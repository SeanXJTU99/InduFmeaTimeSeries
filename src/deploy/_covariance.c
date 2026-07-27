/* Lower-triangular covariance matrix pack/unpack via memcpy.

   Stores a symmetric NxN matrix as N*(N+1)/2 contiguous floats
   (lower triangle, row-major).  Identical layout to the GPU Kalman
   track parameter buffer mC[15] (5x5) / mC[36] (8x8) — enables
   single-DMA-burst transfer on Jetson AGX Orin unified memory.

   Build: included via setup.py Extension('_covariance',
          sources=['src/deploy/_covariance.c'])
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* pack:  NxN row-major float32 matrix -> flat N*(N+1)/2 array        */
/* Each row i contributes (i+1) elements; copied via memcpy.          */
/* ------------------------------------------------------------------ */
static PyObject *
pack_lower_triangular(PyObject *self, PyObject *args)
{
    PyArrayObject *input = NULL;
    npy_intp n, out_len, i, offset;
    float *src, *dst;
    PyArrayObject *result = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input))
        return NULL;

    if (PyArray_NDIM(input) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected 2-D array");
        return NULL;
    }
    if (PyArray_DIMS(input)[0] != PyArray_DIMS(input)[1]) {
        PyErr_Format(PyExc_ValueError,
            "Expected square matrix, got (%ld, %ld)",
            (long)PyArray_DIMS(input)[0], (long)PyArray_DIMS(input)[1]);
        return NULL;
    }
    n = PyArray_DIMS(input)[0];
    if (n < 2 || n > 16) {
        PyErr_Format(PyExc_ValueError,
            "Matrix size %ld out of range [2, 16]", (long)n);
        return NULL;
    }

    /* Input must be float32 and C-contiguous for row-wise memcpy. */
    PyObject *contig = PyArray_FromArray(input, PyArray_DescrFromType(NPY_FLOAT32),
                                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == NULL) return NULL;
    src = (float *)PyArray_DATA((PyArrayObject *)contig);

    out_len = n * (n + 1) / 2;
    result = (PyArrayObject *)PyArray_SimpleNew(1, &out_len, NPY_FLOAT32);
    if (result == NULL) {
        Py_DECREF(contig);
        return NULL;
    }
    dst = (float *)PyArray_DATA(result);

    offset = 0;
    for (i = 0; i < n; i++) {
        memcpy(dst + offset, src + i * n, (size_t)(i + 1) * sizeof(float));
        offset += (npy_intp)(i + 1);
    }

    Py_DECREF(contig);
    return (PyObject *)result;
}


/* ------------------------------------------------------------------ */
/* unpack:  flat N*(N+1)/2 array -> symmetric NxN matrix              */
/* Copies each element to lower + upper triangle.                     */
/* ------------------------------------------------------------------ */
static PyObject *
unpack_lower_triangular(PyObject *self, PyObject *args)
{
    PyArrayObject *input = NULL;
    npy_intp in_len, n, i, j;
    float *src, *dst;
    PyObject *contig = NULL;
    PyArrayObject *result = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input))
        return NULL;

    if (PyArray_NDIM(input) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 1-D packed array");
        return NULL;
    }
    in_len = PyArray_DIMS(input)[0];

    /* Solve n*(n+1)/2 = in_len. */
    for (n = 2; n <= 16; n++) {
        if (n * (n + 1) / 2 == in_len) break;
    }
    if (n > 16) {
        PyErr_Format(PyExc_ValueError,
            "Packed length %ld does not match n*(n+1)/2 for n in [2,16]",
            (long)in_len);
        return NULL;
    }

    contig = PyArray_FromArray(input, PyArray_DescrFromType(NPY_FLOAT32),
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY);
    if (contig == NULL) return NULL;
    src = (float *)PyArray_DATA((PyArrayObject *)contig);

    npy_intp dims[2] = {n, n};
    result = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (result == NULL) {
        Py_DECREF(contig);
        return NULL;
    }
    dst = (float *)PyArray_DATA(result);

    /* Zero-fill then copy lower triangle + mirror. */
    memset(dst, 0, (size_t)n * n * sizeof(float));

    {
        npy_intp flat_idx = 0;
        for (i = 0; i < n; i++) {
            for (j = 0; j <= i; j++) {
                float val = src[flat_idx++];
                dst[i * n + j] = val;
                dst[j * n + i] = val;
            }
        }
    }

    Py_DECREF(contig);
    return (PyObject *)result;
}


/* ------------------------------------------------------------------ */
/* Module                                                              */
/* ------------------------------------------------------------------ */

static PyMethodDef CovarianceMethods[] = {
    {"pack_lower_triangular", pack_lower_triangular, METH_VARARGS,
     "pack_lower_triangular(matrix)\n\n"
     "Pack an NxN symmetric float32 matrix into a 1-D array of\n"
     "N*(N+1)/2 elements (lower triangle, row-major).\n\n"
     "Args:\n"
     "    matrix: 2-D float32 ndarray, shape (N, N).\n\n"
     "Returns:\n"
     "    1-D float32 ndarray, length N*(N+1)/2."},
    {"unpack_lower_triangular", unpack_lower_triangular, METH_VARARGS,
     "unpack_lower_triangular(packed)\n\n"
     "Reconstruct a symmetric NxN float32 matrix from a 1-D\n"
     "lower-triangular packed array.\n\n"
     "Args:\n"
     "    packed: 1-D float32 ndarray, length N*(N+1)/2.\n\n"
     "Returns:\n"
     "    2-D float32 ndarray, shape (N, N)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef covariance_module = {
    PyModuleDef_HEAD_INIT,
    "_covariance",
    "Lower-triangular covariance pack/unpack via memcpy.\n"
    "Matches the GPU Kalman track parameter buffer flat layout.",
    -1,
    CovarianceMethods
};

PyMODINIT_FUNC
PyInit__covariance(void)
{
    import_array();  /* NumPy C API initialisation — required. */
    return PyModule_Create(&covariance_module);
}
