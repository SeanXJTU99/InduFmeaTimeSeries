/* Float32 -> N-bit dictionary quantization via greedy agglomerative clustering.

   Adapted from the approx() compressor algorithm.  Each unique float32 value
   starts as a singleton cluster.  While the cluster count exceeds 2^nbits,
   the two nearest clusters (by value-range gap) are merged.

   The original uses std::map for the distance-to-neighbour lookup; this
   version uses a sorted adjacency array rebuilt each iteration.  For nbits
   <= 12 (max 4096 clusters), the O(K log K) rebuild is acceptable.  Raw
   float32 buffers replace ROOT TTree — zero framework dependency.

   Build:
       setup.py Extension('_compressor', sources=['src/signal/_compressor.c'])
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Cluster: contiguous value range with sample count.                 */
/* ------------------------------------------------------------------ */
typedef struct {
    float min_val;
    float max_val;
    int   count;
    int   active;       /* 1 = alive, 0 = merged away                  */
    int   merged_into;  /* target cluster index when merged (-1 = N/A) */
} Cluster;

/* Comparison: float ascending. */
static int cmp_f(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* Comparison: adjacency entry by distance ascending. */
static int cmp_adj(const void *a, const void *b) {
    float da = ((const struct { float d; int idx; } *)a)->d;
    float db = ((const struct { float d; int idx; } *)b)->d;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

typedef struct { float d; int idx; } AdjEntry;


/* ------------------------------------------------------------------ */
/* approx(data, nbits=8) — main entry point                          */
/* ------------------------------------------------------------------ */
static PyObject *
approx(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *data_arr = NULL;
    int nbits = 8;
    static char *kwlist[] = {"data", "nbits", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|i", kwlist,
                                     &PyArray_Type, &data_arr, &nbits))
        return NULL;

    if (PyArray_NDIM(data_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "data must be 1-D");
        return NULL;
    }
    if (nbits < 2 || nbits > 12) {
        PyErr_SetString(PyExc_ValueError, "nbits must be in [2, 12]");
        return NULL;
    }

    PyObject *contig = PyArray_FromArray(
        data_arr, PyArray_DescrFromType(NPY_FLOAT32),
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == NULL) return NULL;

    npy_intp N = PyArray_DIMS(data_arr)[0];
    float *raw = (float *)PyArray_DATA((PyArrayObject *)contig);
    int max_clusters = 1 << nbits;

    /* ---- empty input ---- */
    if (N == 0) {
        Py_DECREF(contig);
        npy_intp z = 0;
        return Py_BuildValue(
            "{s:O,s:O,s:O,s:f,s:i,s:i,s:i,s:f}",
            "order", PyArray_SimpleNew(1, &z, NPY_UINT16),
            "dict",  PyArray_SimpleNew(1, &z, NPY_FLOAT32),
            "cnt",   PyArray_SimpleNew(1, &z, NPY_INT64),
            "rms_error", 0.0, "nbits", nbits,
            "original_nbytes", 0, "compressed_nbytes", 0, "ratio", 1.0);
    }

    /* ---- Phase 1: find unique values, build initial clusters ---- */

    float *sorted = (float *)malloc((size_t)N * sizeof(float));
    if (sorted == NULL) { Py_DECREF(contig); return PyErr_NoMemory(); }
    memcpy(sorted, raw, (size_t)N * sizeof(float));
    qsort(sorted, (size_t)N, sizeof(float), cmp_f);

    /* Count unique values. */
    int n_uniq = 0, i, j;
    for (i = 0; i < N; i++)
        if (i == 0 || sorted[i] != sorted[i-1]) n_uniq++;

    Cluster *clu = (Cluster *)calloc((size_t)n_uniq, sizeof(Cluster));
    if (clu == NULL) { free(sorted); Py_DECREF(contig); return PyErr_NoMemory(); }

    /* Fill clusters. */
    {
        int ci = 0;
        for (i = 0; i < N; ) {
            float v = sorted[i];
            j = i;
            while (j < N && sorted[j] == v) j++;
            clu[ci].min_val = v;
            clu[ci].max_val = v;
            clu[ci].count   = j - i;
            clu[ci].active  = 1;
            clu[ci].merged_into = -1;
            ci++;
            i = j;
        }
    }

    int n_active = n_uniq;

    /* ---- Phase 2: greedy merge ---- */

    AdjEntry *adj = NULL;
    int n_adj = 0;

    while (n_active > max_clusters && n_active >= 2) {

        /* Build adjacency: each adjacent pair of active clusters. */
        {
            int prev = -1;
            n_adj = 0;
            for (i = 0; i < n_uniq; i++) {
                if (!clu[i].active) continue;
                if (prev >= 0) {
                    if (n_adj == 0) {
                        adj = (AdjEntry *)malloc(
                            (size_t)(n_active - 1) * sizeof(AdjEntry));
                        if (adj == NULL) goto nomem;
                    }
                    adj[n_adj].d   = clu[i].min_val - clu[prev].max_val;
                    adj[n_adj].idx = prev;
                    n_adj++;
                }
                prev = i;
            }
            if (n_adj == 0) break; /* single cluster left — done */
        }

        /* Find the minimum-distance pair. */
        int best = 0;
        for (i = 1; i < n_adj; i++)
            if (adj[i].d < adj[best].d) best = i;

        int left = adj[best].idx;

        /* Find right neighbour: next active cluster after left. */
        int right = -1;
        for (i = left + 1; i < n_uniq; i++) {
            if (clu[i].active) { right = i; break; }
        }
        if (right < 0) { free(adj); adj = NULL; break; }

        /* Merge right into left. */
        clu[left].max_val = clu[right].max_val;
        clu[left].count  += clu[right].count;
        clu[right].active = 0;
        clu[right].merged_into = left;
        n_active--;

        free(adj);
        adj = NULL;
    }

    free(adj);

    /* ---- Phase 3: map each original sample to its surviving cluster ---- */

    /* For each unique value, find the root cluster after all merges. */
    int *uniq_to_new = (int *)malloc((size_t)n_uniq * sizeof(int));
    if (uniq_to_new == NULL) goto nomem;

    int n_survive = 0;
    int *orig_map = (int *)malloc((size_t)n_uniq * sizeof(int));
    if (orig_map == NULL) { free(uniq_to_new); goto nomem; }

    for (i = 0; i < n_uniq; i++) {
        int root = i;
        while (!clu[root].active && clu[root].merged_into >= 0)
            root = clu[root].merged_into;
        if (clu[root].active) {
            /* Assign a new index to this root cluster. */
            int found = -1;
            for (j = 0; j < n_survive; j++)
                if (orig_map[j] == root) { found = j; break; }
            if (found < 0) {
                orig_map[n_survive] = root;
                found = n_survive++;
            }
            uniq_to_new[i] = found;
        } else {
            uniq_to_new[i] = -1;
        }
    }

    /* ---- Phase 4: build output arrays ---- */

    npy_intp olen = N;
    PyArrayObject *order_arr = (PyArrayObject *)PyArray_SimpleNew(
        1, &olen, (nbits <= 8) ? NPY_UINT8 : NPY_UINT16);
    if (order_arr == NULL) { free(orig_map); free(uniq_to_new); goto nomem; }

    npy_intp dlen = n_survive;
    PyArrayObject *dict_arr = (PyArrayObject *)PyArray_SimpleNew(
        1, &dlen, NPY_FLOAT32);
    PyArrayObject *cnt_arr = (PyArrayObject *)PyArray_SimpleNew(
        1, &dlen, NPY_INT64);
    if (dict_arr == NULL || cnt_arr == NULL) {
        Py_XDECREF(dict_arr); Py_XDECREF(cnt_arr);
        Py_DECREF(order_arr); free(orig_map); free(uniq_to_new); goto nomem;
    }

    float  *dict_data = (float *)PyArray_DATA(dict_arr);
    int64_t *cnt_data = (int64_t *)PyArray_DATA(cnt_arr);

    for (i = 0; i < n_survive; i++) {
        int root = orig_map[i];
        dict_data[i] = (clu[root].min_val + clu[root].max_val) * 0.5f;
        cnt_data[i]  = (int64_t)clu[root].count;
    }

    /* Map each raw sample: binary-search its value in sorted[], find the
       unique-val index, walk to root, look up output index. */
    double sqsum = 0.0;
    for (i = 0; i < N; i++) {
        float val = raw[i];

        /* Binary search in sorted. */
        float *hit = (float *)bsearch(&val, sorted, (size_t)N, sizeof(float), cmp_f);
        if (hit == NULL) { hit = sorted; } /* shouldn't happen */

        /* Determine the unique-value index: count how many distinct values
           precede this position. */
        int u_idx = 0;
        float *p = sorted;
        float prev_val = *p;
        while (p < hit) {
            if (*p != prev_val) { prev_val = *p; u_idx++; }
            p++;
        }

        if (u_idx >= n_uniq) u_idx = n_uniq - 1;
        int new_idx = uniq_to_new[u_idx];
        if (new_idx < 0) new_idx = 0;

        if (nbits <= 8)
            *(uint8_t  *)PyArray_GETPTR1(order_arr, i) = (uint8_t)new_idx;
        else
            *(uint16_t *)PyArray_GETPTR1(order_arr, i) = (uint16_t)new_idx;

        float delta = val - dict_data[new_idx];
        sqsum += (double)delta * delta;
    }

    double rms = sqrt(sqsum / (double)N);
    int idx_bytes = (int)ceil((double)N * nbits / 8.0);
    int d_bytes   = (int)(n_survive * 4);
    int c_bytes   = (int)(n_survive * 4);
    int comp_bytes = idx_bytes + d_bytes + c_bytes;
    int orig_bytes = (int)(N * 4);
    double ratio = (double)comp_bytes / (double)orig_bytes;

    free(orig_map);
    free(uniq_to_new);
    free(clu);
    free(sorted);
    Py_DECREF(contig);

    return Py_BuildValue(
        "{s:O,s:O,s:O,s:f,s:i,s:i,s:i,s:f}",
        "order", order_arr,
        "dict", dict_arr,
        "cnt", cnt_arr,
        "rms_error", rms,
        "nbits", nbits,
        "original_nbytes", orig_bytes,
        "compressed_nbytes", comp_bytes,
        "ratio", ratio);

nomem:
    free(adj);
    free(clu);
    free(sorted);
    Py_DECREF(contig);
    return PyErr_NoMemory();
}


/* ------------------------------------------------------------------ */
/* Module                                                              */
/* ------------------------------------------------------------------ */

static PyMethodDef CompressorMethods[] = {
    {"approx", (PyCFunction)approx, METH_VARARGS | METH_KEYWORDS,
     "approx(data, nbits=8)\n\n"
     "Compress float32 data via N-bit dictionary quantization using\n"
     "greedy agglomerative clustering.\n\n"
     "Args:\n"
     "    data: 1-D float32 ndarray.\n"
     "    nbits: bit width (2-12, default 8).\n\n"
     "Returns:\n"
     "    dict with keys: order, dict, cnt, rms_error, nbits,\n"
     "    original_nbytes, compressed_nbytes, ratio."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef compressor_module = {
    PyModuleDef_HEAD_INIT,
    "_compressor",
    "Float32 -> N-bit dictionary quantization via greedy agglomerative\n"
    "clustering.  Adapted from the approx() algorithm.",
    -1,
    CompressorMethods
};

PyMODINIT_FUNC
PyInit__compressor(void)
{
    import_array();
    return PyModule_Create(&compressor_module);
}
