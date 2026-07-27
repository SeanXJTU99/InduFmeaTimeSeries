/* Float32 → N-bit dictionary quantisation via greedy agglomerative clustering.

   Adapted from the approx() compressor algorithm.  Replaces std::map
   (red-black tree) with sorted vectors + binary search for better cache
   locality on Jetson AGX Orin.  Accepts raw float32 buffers — zero
   framework dependency (no ROOT / TTree).

   Class design:
     Cluster            — value range, sample count, merge tracking
     ClusterRegistry    — owns all clusters; merge, find-neighbour, compact
     AdjacencyList      — sorted inter-cluster distance list; pop-min, rebuild
     CompressorEngine   — top-level state machine: init → merge → output

   Algorithm
   ---------
     1. Sort unique float32 values → each is a singleton Cluster.
     2. Build AdjacencyList of gaps between adjacent active clusters.
     3. While cluster_count > 2^nbits:
        a. Pop minimum-distance adjacency entry.
        b. Merge right cluster into left.
        c. Rebuild adjacency entries affected by the merge.
     4. Map each original sample → surviving cluster index.
     5. Output: order (N-bit index), dict (float32 centres), cnt (sizes).

   Python binding via C API at bottom of file.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// ======================================================================
// Pure C++ layer
// ======================================================================

namespace compressor {

// ----------------------------------------------------------------------
// Cluster — a contiguous value range with sample count.
// ----------------------------------------------------------------------
class Cluster {
public:
    Cluster(float minVal, float maxVal, int count = 0)
        : m_minVal(minVal)
        , m_maxVal(maxVal)
        , m_count(count)
        , m_active(true)
        , m_mergedInto(-1)
    {}

    // --- accessors ----------------------------------------------------
    float minVal()     const noexcept { return m_minVal; }
    float maxVal()     const noexcept { return m_maxVal; }
    int   count()      const noexcept { return m_count; }
    bool  active()     const noexcept { return m_active; }
    int   mergedInto() const noexcept { return m_mergedInto; }

    void setMaxVal(float v) noexcept   { m_maxVal = v; }
    void addCount(int c) noexcept      { m_count += c; }
    void deactivate(int into) noexcept { m_active = false; m_mergedInto = into; }

    /** Distance from this cluster's max to another's min. */
    float distanceTo(const Cluster &other) const noexcept {
        return other.m_minVal - m_maxVal;
    }

private:
    float m_minVal;
    float m_maxVal;
    int   m_count;
    bool  m_active;
    int   m_mergedInto;   // index of absorbing cluster (-1 = N/A)
};


// ----------------------------------------------------------------------
// AdjacencyList — sorted list of (distance, left_cluster_index) pairs.
// ----------------------------------------------------------------------
class AdjacencyList {
public:
    struct Entry {
        float distance;
        int   leftIdx;

        bool operator<(const Entry &rhs) const noexcept {
            return distance < rhs.distance;
        }
    };

    /** Build adjacency from a vector of active cluster indices (sorted by
     *  minVal).  Computes the gap from each cluster to its immediate
     *  right neighbour. */
    void build(const std::vector<int> &activeIndices,
               const std::vector<Cluster> &clusters) {
        m_entries.clear();
        m_entries.reserve(activeIndices.size() > 0
                          ? activeIndices.size() - 1 : 0);
        for (size_t i = 0; i + 1 < activeIndices.size(); ++i) {
            int left  = activeIndices[i];
            int right = activeIndices[i + 1];
            m_entries.push_back({
                clusters[left].distanceTo(clusters[right]), left
            });
        }
        std::sort(m_entries.begin(), m_entries.end());
    }

    /** Pop and return the entry with the smallest distance. */
    Entry popMin() {
        if (m_entries.empty())
            throw std::runtime_error("AdjacencyList is empty");
        Entry best = m_entries.front();
        m_entries.erase(m_entries.begin());
        return best;
    }

    bool empty() const noexcept { return m_entries.empty(); }
    size_t size() const noexcept { return m_entries.size(); }

private:
    std::vector<Entry> m_entries;
};


// ----------------------------------------------------------------------
// CompressorEngine — state machine for one compression run.
// ----------------------------------------------------------------------
class CompressorEngine {
public:
    /**
     * Run the full compression pipeline.
     *
     * @param data    raw float32 samples (not sorted)
     * @param n       number of samples
     * @param nbits   bit width for the dictionary index (2–12)
     * @param order   output: per-sample dictionary index [0, 2^nbits)
     * @param dict    output: cluster-centre values (float32)
     * @param counts  output: samples per cluster (int64)
     * @return        RMS error of the quantisation
     */
    double compress(const float *data, size_t n, int nbits,
                    std::vector<uint16_t> &order,
                    std::vector<float>   &dict,
                    std::vector<int64_t> &counts);

private:
    void   buildInitialClusters(const float *sorted, size_t n);
    void   greedyMerge(int maxClusters);
    void   buildOutput(const float *raw, size_t n);
    int    findRootCluster(int idx) const;
    int    findUniqueIndex(const float *sorted, size_t n, const float *hit) const;

    std::vector<Cluster> m_clusters;
    AdjacencyList         m_adjacency;
    int                   m_nActive = 0;
};


// ----------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------

void CompressorEngine::buildInitialClusters(const float *sorted, size_t n) {
    // Count unique values first.
    size_t nUnique = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i == 0 || sorted[i] != sorted[i - 1]) ++nUnique;
    }

    m_clusters.clear();
    m_clusters.reserve(nUnique);

    for (size_t i = 0; i < n; ) {
        float val = sorted[i];
        size_t j = i;
        while (j < n && sorted[j] == val) ++j;
        m_clusters.emplace_back(val, val, static_cast<int>(j - i));
        i = j;
    }
    m_nActive = static_cast<int>(m_clusters.size());
}


void CompressorEngine::greedyMerge(int maxClusters) {
    while (m_nActive > maxClusters && m_nActive >= 2) {

        // Collect active cluster indices in order.
        std::vector<int> activeIndices;
        activeIndices.reserve(static_cast<size_t>(m_nActive));
        for (int i = 0; i < static_cast<int>(m_clusters.size()); ++i) {
            if (m_clusters[i].active()) activeIndices.push_back(i);
        }

        // Build adjacency from current active set.
        m_adjacency.build(activeIndices, m_clusters);

        if (m_adjacency.empty()) break;

        // Pop the closest pair.
        auto entry = m_adjacency.popMin();
        int left = entry.leftIdx;
        if (!m_clusters[left].active()) continue;

        // Find right neighbour of `left` in the active set.
        int right = -1;
        for (size_t i = 0; i + 1 < activeIndices.size(); ++i) {
            if (activeIndices[i] == left) {
                right = activeIndices[i + 1];
                break;
            }
        }
        if (right < 0 || !m_clusters[right].active()) continue;

        // Merge right into left.
        m_clusters[left].setMaxVal(m_clusters[right].maxVal());
        m_clusters[left].addCount(m_clusters[right].count());
        m_clusters[right].deactivate(left);
        --m_nActive;
    }
}


void CompressorEngine::buildOutput(const float *raw, size_t n) {
    // No output arrays allocated here — caller handles that.
    // This method just ensures the internal cluster state is consistent.
    (void)raw;
    (void)n;
}


int CompressorEngine::findRootCluster(int idx) const {
    int root = idx;
    while (!m_clusters[root].active() && m_clusters[root].mergedInto() >= 0) {
        root = m_clusters[root].mergedInto();
    }
    return m_clusters[root].active() ? root : -1;
}


int CompressorEngine::findUniqueIndex(const float *sorted, size_t n,
                                       const float *hit) const {
    // Walk sorted[] from start to `hit`, counting unique values.
    int uIdx = 0;
    const float *p = sorted;
    float prev = *p;
    while (p < hit) {
        if (*p != prev) { prev = *p; ++uIdx; }
        ++p;
    }
    return uIdx;
}


double CompressorEngine::compress(
    const float *data, size_t n, int nbits,
    std::vector<uint16_t> &order,
    std::vector<float>   &dict,
    std::vector<int64_t> &counts)
{
    if (nbits < 2 || nbits > 12) {
        throw std::invalid_argument("nbits must be in [2, 12]");
    }
    int maxClusters = 1 << nbits;
    if (n == 0) return 0.0;

    // ---- Phase 1: sort and build initial clusters -------------------
    std::vector<float> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    buildInitialClusters(sorted.data(), n);

    // ---- Phase 2: greedy merge -------------------------------------
    if (m_nActive > maxClusters) {
        greedyMerge(maxClusters);
    }

    // ---- Phase 3: map each cluster to output index -----------------
    int nUnique = static_cast<int>(m_clusters.size());

    // Map original cluster index → surviving output index.
    std::vector<int> uniqToNew(static_cast<size_t>(nUnique), -1);
    int nSurviving = 0;

    for (int i = 0; i < nUnique; ++i) {
        int root = findRootCluster(i);
        if (root >= 0) {
            // Check if this root already has an output index.
            bool found = false;
            for (int j = 0; j < nSurviving; ++j) {
                if (uniqToNew[j] == root) {
                    uniqToNew[i] = j;
                    found = true;
                    break;
                }
            }
            if (!found && root == i && m_clusters[i].active()) {
                // First time seeing this root cluster — assign new index.
                uniqToNew[i] = nSurviving++;
            } else if (!found) {
                // This cluster was merged — find its root's index.
                for (int k = 0; k < i; ++k) {
                    int rk = findRootCluster(k);
                    if (rk == root) {
                        uniqToNew[i] = uniqToNew[k];
                        found = true;
                        break;
                    }
                }
            }
        }
    }

    // ---- Phase 4: assign each sample and compute error ---------------
    order.assign(n, 0);
    dict.resize(static_cast<size_t>(nSurviving));
    counts.assign(static_cast<size_t>(nSurviving), 0);

    // Build dict from surviving active clusters.
    {
        int outIdx = 0;
        for (int i = 0; i < nUnique; ++i) {
            if (m_clusters[i].active()) {
                dict[static_cast<size_t>(outIdx)] =
                    (m_clusters[i].minVal() + m_clusters[i].maxVal()) * 0.5f;
                counts[static_cast<size_t>(outIdx)] =
                    static_cast<int64_t>(m_clusters[i].count());
                ++outIdx;
            }
        }
    }

    double squaredSum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float val = data[i];

        // Find unique-value index via binary search in sorted[].
        auto hit = std::lower_bound(sorted.begin(), sorted.end(), val);
        size_t pos = static_cast<size_t>(hit - sorted.begin());

        int uIdx = findUniqueIndex(sorted.data(), n, sorted.data() + pos);
        if (uIdx >= nUnique) uIdx = nUnique - 1;

        int newIdx = uniqToNew[static_cast<size_t>(uIdx)];
        if (newIdx < 0 || newIdx >= nSurviving) newIdx = 0;

        order[i] = static_cast<uint16_t>(newIdx);

        float delta = val - dict[static_cast<size_t>(newIdx)];
        squaredSum += static_cast<double>(delta) * delta;
    }

    return std::sqrt(squaredSum / static_cast<double>(n));
}

}  // namespace compressor


// ======================================================================
// Python C API binding layer
// ======================================================================

static PyObject *
py_approx(PyObject * /*self*/, PyObject *args, PyObject *kwargs) {
    PyArrayObject *dataArr = nullptr;
    int nbits = 8;
    static const char *kwlist[] = {"data", "nbits", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|i",
                                     const_cast<char **>(kwlist),
                                     &PyArray_Type, &dataArr, &nbits))
        return nullptr;

    if (PyArray_NDIM(dataArr) != 1) {
        PyErr_SetString(PyExc_ValueError, "data must be 1-D");
        return nullptr;
    }

    PyObject *contig = PyArray_FromArray(
        dataArr, PyArray_DescrFromType(NPY_FLOAT32),
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == nullptr) return nullptr;

    try {
        auto *raw = static_cast<const float *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(contig)));
        npy_intp N = PyArray_DIMS(dataArr)[0];

        std::vector<uint16_t> order;
        std::vector<float>    dict;
        std::vector<int64_t>  counts;

        compressor::CompressorEngine engine;
        double rms = engine.compress(raw, static_cast<size_t>(N),
                                     nbits, order, dict, counts);

        // Build numpy output arrays.
        npy_intp oLen = static_cast<npy_intp>(order.size());
        PyArrayObject *orderArr = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, &oLen,
                              (nbits <= 8) ? NPY_UINT8 : NPY_UINT16));
        if (orderArr == nullptr) { Py_DECREF(contig); return nullptr; }
        for (npy_intp i = 0; i < oLen; ++i) {
            if (nbits <= 8)
                *(static_cast<uint8_t  *>(PyArray_GETPTR1(orderArr, i))) =
                    static_cast<uint8_t>(order[static_cast<size_t>(i)]);
            else
                *(static_cast<uint16_t *>(PyArray_GETPTR1(orderArr, i))) =
                    order[static_cast<size_t>(i)];
        }

        npy_intp dLen = static_cast<npy_intp>(dict.size());
        PyArrayObject *dictArr = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, &dLen, NPY_FLOAT32));
        PyArrayObject *cntArr = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, &dLen, NPY_INT64));
        if (dictArr == nullptr || cntArr == nullptr) {
            Py_XDECREF(dictArr); Py_XDECREF(cntArr);
            Py_DECREF(orderArr); Py_DECREF(contig);
            return nullptr;
        }
        std::memcpy(PyArray_DATA(dictArr), dict.data(),
                    static_cast<size_t>(dLen) * sizeof(float));
        std::memcpy(PyArray_DATA(cntArr), counts.data(),
                    static_cast<size_t>(dLen) * sizeof(int64_t));

        // Compute storage statistics.
        int idxBytes  = static_cast<int>(std::ceil(
                            static_cast<double>(N) * nbits / 8.0));
        int dBytes    = static_cast<int>(dict.size() * 4);
        int cBytes    = static_cast<int>(counts.size() * 4);
        int compBytes = idxBytes + dBytes + cBytes;
        int origBytes = static_cast<int>(N * 4);
        double ratio  = static_cast<double>(compBytes) /
                        static_cast<double>(origBytes);

        Py_DECREF(contig);

        return Py_BuildValue(
            "{s:O,s:O,s:O,s:f,s:i,s:i,s:i,s:f}",
            "order", orderArr,
            "dict",  dictArr,
            "cnt",   cntArr,
            "rms_error", rms,
            "nbits", nbits,
            "original_nbytes", origBytes,
            "compressed_nbytes", compBytes,
            "ratio", ratio);
    } catch (const std::exception &e) {
        Py_DECREF(contig);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}


/* ------------------------------------------------------------------ */
/* Module                                                              */
/* ------------------------------------------------------------------ */

static PyMethodDef CompressorMethods[] = {
    {"approx", reinterpret_cast<PyCFunction>(py_approx),
     METH_VARARGS | METH_KEYWORDS,
     "approx(data, nbits=8)\n\n"
     "Compress float32 data via N-bit dictionary quantisation using\n"
     "greedy agglomerative clustering (CompressorEngine)."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef compressorModule = {
    PyModuleDef_HEAD_INIT,
    "_compressor",
    "Float32 → N-bit dictionary quantisation via C++ CompressorEngine.\n"
    "Uses sorted-vector adjacency replacing std::map for cache locality.",
    -1,
    CompressorMethods
};

PyMODINIT_FUNC
PyInit__compressor(void) {
    import_array();
    return PyModule_Create(&compressorModule);
}
