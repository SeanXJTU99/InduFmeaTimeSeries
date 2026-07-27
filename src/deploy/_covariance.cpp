/* Lower-triangular symmetric covariance matrix packer.

   Stores N×N symmetric matrices as N*(N+1)/2 contiguous floats
   (lower triangle, row-major).  Matches the GPU Kalman track
   parameter buffer layout mC[15] (5×5) / mC[36] (8×8), enabling
   single-DMA-burst transfer on Jetson AGX Orin unified memory.

   Class design:
     CovariancePacker<N>  — template with compile-time loop unrolling
     CovariancePackerBase — virtual base for runtime dispatch

   Python binding via C API at bottom of file.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>

// ======================================================================
// Pure C++ layer — no Python dependency
// ======================================================================

namespace covariance {

/**
 * Abstract base for runtime-polymorphic pack/unpack.
 */
class CovariancePackerBase {
public:
    virtual ~CovariancePackerBase() = default;

    /** @return number of elements in the packed lower-triangular array. */
    virtual int packedSize() const noexcept = 0;

    /** @return matrix dimension N. */
    virtual int dimension() const noexcept = 0;

    /**
     * Pack an N×N row-major matrix into a lower-triangular flat array.
     * @param src  N×N float32 matrix, row-major contiguous.
     * @param dst  pre-allocated buffer of packedSize() floats.
     */
    virtual void pack(const float *src, float *dst) const noexcept = 0;

    /**
     * Unpack a lower-triangular flat array into a symmetric N×N matrix.
     * @param src  packed array of packedSize() floats.
     * @param dst  pre-allocated N×N float32 buffer (zero-filled).
     */
    virtual void unpack(const float *src, float *dst) const noexcept = 0;
};


/**
 * Compile-time-specialised packer for a fixed dimension N.
 *
 * The template parameter N triggers loop unrolling by the compiler
 * (all loop bounds are compile-time constants).  Used for the common
 * 5×5 (Kalman state) and 8×8 (extended Kalman) cases.
 */
template <int N>
class CovariancePacker : public CovariancePackerBase {
    static_assert(N >= 2 && N <= 16, "N must be in [2, 16]");

public:
    static constexpr int kPackedSize = N * (N + 1) / 2;

    int packedSize() const noexcept override { return kPackedSize; }
    int dimension()  const noexcept override { return N; }

    void pack(const float *src, float *dst) const noexcept override {
        int offset = 0;
        for (int i = 0; i < N; ++i) {
            std::memcpy(dst + offset, src + i * N,
                        static_cast<size_t>(i + 1) * sizeof(float));
            offset += i + 1;
        }
    }

    void unpack(const float *src, float *dst) const noexcept override {
        std::memset(dst, 0, static_cast<size_t>(N * N) * sizeof(float));
        int flat = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                float val = src[flat++];
                dst[i * N + j] = val;
                dst[j * N + i] = val;
            }
        }
    }
};


/**
 * Factory: create a packer for the given packed-array length.
 *
 * Solves N*(N+1)/2 = packedLen for integer N.
 */
std::unique_ptr<CovariancePackerBase>
makePacker(int packedLen) {
    for (int n = 2; n <= 16; ++n) {
        if (n * (n + 1) / 2 == packedLen) {
            switch (n) {
                case  2: return std::make_unique<CovariancePacker< 2>>();
                case  3: return std::make_unique<CovariancePacker< 3>>();
                case  4: return std::make_unique<CovariancePacker< 4>>();
                case  5: return std::make_unique<CovariancePacker< 5>>();
                case  6: return std::make_unique<CovariancePacker< 6>>();
                case  7: return std::make_unique<CovariancePacker< 7>>();
                case  8: return std::make_unique<CovariancePacker< 8>>();
                case  9: return std::make_unique<CovariancePacker< 9>>();
                case 10: return std::make_unique<CovariancePacker<10>>();
                case 11: return std::make_unique<CovariancePacker<11>>();
                case 12: return std::make_unique<CovariancePacker<12>>();
                case 13: return std::make_unique<CovariancePacker<13>>();
                case 14: return std::make_unique<CovariancePacker<14>>();
                case 15: return std::make_unique<CovariancePacker<15>>();
                case 16: return std::make_unique<CovariancePacker<16>>();
            }
        }
    }
    throw std::invalid_argument(
        "packedLen does not match N*(N+1)/2 for N in [2,16]");
}

/**
 * Factory: create a packer for an N×N square matrix.
 */
std::unique_ptr<CovariancePackerBase>
makePackerForMatrix(int n) {
    switch (n) {
        case  2: return std::make_unique<CovariancePacker< 2>>();
        case  3: return std::make_unique<CovariancePacker< 3>>();
        case  4: return std::make_unique<CovariancePacker< 4>>();
        case  5: return std::make_unique<CovariancePacker< 5>>();
        case  6: return std::make_unique<CovariancePacker< 6>>();
        case  7: return std::make_unique<CovariancePacker< 7>>();
        case  8: return std::make_unique<CovariancePacker< 8>>();
        case  9: return std::make_unique<CovariancePacker< 9>>();
        case 10: return std::make_unique<CovariancePacker<10>>();
        case 11: return std::make_unique<CovariancePacker<11>>();
        case 12: return std::make_unique<CovariancePacker<12>>();
        case 13: return std::make_unique<CovariancePacker<13>>();
        case 14: return std::make_unique<CovariancePacker<14>>();
        case 15: return std::make_unique<CovariancePacker<15>>();
        case 16: return std::make_unique<CovariancePacker<16>>();
    }
    throw std::invalid_argument("Matrix dimension out of range [2, 16]");
}

}  // namespace covariance


// ======================================================================
// Python C API binding layer
// ======================================================================

namespace {

/**
 * Helper: extract a contiguous float32 2-D numpy array, validate it is
 * square, and return its dimension and data pointer.
 */
int validateSquareMatrix(PyArrayObject *arr, const float *&data) {
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected 2-D array");
        return -1;
    }
    int n = static_cast<int>(PyArray_DIMS(arr)[0]);
    if (n != static_cast<int>(PyArray_DIMS(arr)[1])) {
        PyErr_Format(PyExc_ValueError,
                     "Expected square matrix, got (%ld, %ld)",
                     (long)PyArray_DIMS(arr)[0],
                     (long)PyArray_DIMS(arr)[1]);
        return -1;
    }
    if (n < 2 || n > 16) {
        PyErr_Format(PyExc_ValueError,
                     "Matrix size %d out of range [2, 16]", n);
        return -1;
    }
    // Ensure float32 contiguous.
    PyObject *contig = PyArray_FromArray(
        arr, PyArray_DescrFromType(NPY_FLOAT32),
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == nullptr) return -1;
    data = static_cast<const float *>(PyArray_DATA(
        reinterpret_cast<PyArrayObject *>(contig)));
    // Note: contig is leaked if we don't track it.  We'll re-fetch
    // via PyArray_FromArray inside the pack/unpack functions where
    // we need the data, keeping the reference local.
    Py_DECREF(contig);
    return n;
}

}  // anonymous namespace


/* ------------------------------------------------------------------ */
/* pack_lower_triangular(matrix) -> 1-D float32 ndarray               */
/* ------------------------------------------------------------------ */
static PyObject *
py_pack_lower_triangular(PyObject * /*self*/, PyObject *args) {
    PyArrayObject *input = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input))
        return nullptr;

    if (PyArray_NDIM(input) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected 2-D array");
        return nullptr;
    }
    int n = static_cast<int>(PyArray_DIMS(input)[0]);
    if (n != static_cast<int>(PyArray_DIMS(input)[1])) {
        PyErr_SetString(PyExc_ValueError, "Expected square matrix");
        return nullptr;
    }

    // Ensure float32 C-contiguous.
    PyObject *contig = PyArray_FromArray(
        input, PyArray_DescrFromType(NPY_FLOAT32),
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == nullptr) return nullptr;

    try {
        auto packer = covariance::makePackerForMatrix(n);
        const float *src = static_cast<const float *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(contig)));

        npy_intp outLen = packer->packedSize();
        PyArrayObject *result = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, &outLen, NPY_FLOAT32));
        if (result == nullptr) {
            Py_DECREF(contig);
            return nullptr;
        }

        float *dst = static_cast<float *>(PyArray_DATA(result));
        packer->pack(src, dst);

        Py_DECREF(contig);
        return reinterpret_cast<PyObject *>(result);
    } catch (const std::exception &e) {
        Py_DECREF(contig);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}


/* ------------------------------------------------------------------ */
/* unpack_lower_triangular(packed) -> 2-D (N,N) float32 ndarray       */
/* ------------------------------------------------------------------ */
static PyObject *
py_unpack_lower_triangular(PyObject * /*self*/, PyObject *args) {
    PyArrayObject *input = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input))
        return nullptr;

    if (PyArray_NDIM(input) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 1-D packed array");
        return nullptr;
    }

    PyObject *contig = PyArray_FromArray(
        input, PyArray_DescrFromType(NPY_FLOAT32),
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == nullptr) return nullptr;

    try {
        int packedLen = static_cast<int>(PyArray_DIMS(input)[0]);
        auto packer = covariance::makePacker(packedLen);
        const float *src = static_cast<const float *>(
            PyArray_DATA(reinterpret_cast<PyArrayObject *>(contig)));

        npy_intp dims[2] = {static_cast<npy_intp>(packer->dimension()),
                            static_cast<npy_intp>(packer->dimension())};
        PyArrayObject *result = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(2, dims, NPY_FLOAT32));
        if (result == nullptr) {
            Py_DECREF(contig);
            return nullptr;
        }

        float *dst = static_cast<float *>(PyArray_DATA(result));
        packer->unpack(src, dst);

        Py_DECREF(contig);
        return reinterpret_cast<PyObject *>(result);
    } catch (const std::exception &e) {
        Py_DECREF(contig);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}


/* ------------------------------------------------------------------ */
/* Module definition                                                   */
/* ------------------------------------------------------------------ */

static PyMethodDef CovarianceMethods[] = {
    {"pack_lower_triangular",   py_pack_lower_triangular, METH_VARARGS,
     "pack_lower_triangular(matrix)\n\n"
     "Pack an N×N symmetric float32 matrix into a 1-D flat array of\n"
     "N*(N+1)/2 elements (lower triangle, row-major)."},
    {"unpack_lower_triangular", py_unpack_lower_triangular, METH_VARARGS,
     "unpack_lower_triangular(packed)\n\n"
     "Reconstruct a symmetric N×N float32 matrix from a 1-D\n"
     "lower-triangular packed array."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef covariance_module = {
    PyModuleDef_HEAD_INIT,
    "_covariance",
    "Lower-triangular covariance pack/unpack via C++ CovariancePacker<N>.\n"
    "Template-specialised for N ∈ [2, 16] with compile-time loop unrolling.",
    -1,
    CovarianceMethods
};

PyMODINIT_FUNC
PyInit__covariance(void) {
    import_array();
    return PyModule_Create(&covariance_module);
}
