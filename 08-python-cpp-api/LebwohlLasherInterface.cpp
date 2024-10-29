#include "LebwohlLasher.cpp"

// #define PY_SSIZE_T_CLEAN // Needed if not on Python 3.13
#include <Python.h>

// Python Wrappers for the function:
// bool LebwohlLasher(std::string program, int nsteps, int nmax, double temp, bool pflag)

// https://docs.python.org/3/extending/extending.html

extern "C" {

bool py_arr_to_vector(PyObject* py_arr, std::vector< std::vector< double > > &arr, int nmax) 
{
    for (int i = 0; i < nmax; i++) {
        PyObject* row = PyList_GetItem(py_arr, i);
        // Check if row is a list 
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
            return false;
        }
        for (int j = 0; j < nmax; ++j) {
            PyObject* item = PyList_GetItem(row, j);
            // Check if item is a float
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Expected a list of lists of floats");
                return false;
            }
            // Populate the C++ vector
            arr[i][j] = PyFloat_AsDouble(item);
        }
    }
    return true;
}

static PyObject* py_LebwohlLasher(PyObject* self, PyObject* args) 
{
    const char* program;
    int nsteps;
    int nmax;
    double Ts;
    int threadcount;

    if (!PyArg_ParseTuple(args, "siidi", &program, &nsteps, &nmax, &Ts, &threadcount)) {
        return NULL;
    }

    std::vector< std::vector< double > > initial_lattice(nmax, std::vector< double >(nmax, 0.0));
    initdat(initial_lattice, nmax, 2*M_PI);
    std::vector< std::vector< double > > final_lattice = initial_lattice;

    LebwohlLasher(final_lattice, program, nsteps, nmax, Ts, threadcount);

    // Convert initial_lattice to a Python list of lists
    PyObject* py_initial_lattice = PyList_New(nmax);
    for (int i = 0; i < nmax; ++i) {
        PyObject* row = PyList_New(nmax);
        for (int j = 0; j < nmax; ++j) {
            PyList_SetItem(row, j, PyFloat_FromDouble(initial_lattice[i][j]));
        }
        PyList_SetItem(py_initial_lattice, i, row);
    }

    // Convert final_lattice to a Python list of lists
    PyObject* py_final_lattice = PyList_New(nmax);
    for (int i = 0; i < nmax; ++i) {
        PyObject* row = PyList_New(nmax);
        for (int j = 0; j < nmax; ++j) {
            PyList_SetItem(row, j, PyFloat_FromDouble(final_lattice[i][j]));
        }
        PyList_SetItem(py_final_lattice, i, row);
    }

    // Return a tuple containing the result, initial_lattice, and final_lattice
    return Py_BuildValue("OO", py_initial_lattice, py_final_lattice);
}

static PyObject* py_energies(PyObject* self, PyObject* args) 
{
    // double one_energy(const std::vector< std::vector< double > > &arr , int ix, int iy, int nmax) 
    PyObject* py_arr;
    int nmax;

    if (!PyArg_ParseTuple(args, "Oi", &py_arr, &nmax)) {
        return NULL;
    }

    // Check if py_arr is a list
    if (!PyList_Check(py_arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    std::vector< std::vector< double > > arr(nmax, std::vector< double >(nmax, 0.0));
    for (int i = 0; i < nmax; ++i) {
        PyObject* row = PyList_GetItem(py_arr, i);
        // Check if row is a list
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
            return NULL;
        }
        
        for (int j = 0; j < nmax; ++j) {
            PyObject* item = PyList_GetItem(row, j);
            // Check if item is a float
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Expected a list of lists of floats");
                return NULL;
            }
            arr[i][j] = PyFloat_AsDouble(item);
        }
    }

    std::vector< std::vector< double > > result(nmax, std::vector< double >(nmax, 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nmax; ++i) 
    {
        for (int j = 0; j < nmax; ++j) 
        {
            result[i][j] = one_energy(arr, i, j, nmax);
        }
    }

    PyObject* py_result = PyList_New(nmax);
    for (int i = 0; i < nmax; ++i) {
        PyObject* row = PyList_New(nmax);
        for (int j = 0; j < nmax; ++j) {
            PyList_SetItem(row, j, PyFloat_FromDouble(result[i][j]));
        }
        PyList_SetItem(py_result, i, row);
    }

    return Py_BuildValue("O", py_result);
}

static PyObject* py_initdat(PyObject* self, PyObject* args) 
{
    // void initdat(std::vector< std::vector< double > > &arr, int nmax, double scale) 
    int nmax;
    int nthreads;
    
    if (!PyArg_ParseTuple(args, "ii", &nmax, &nthreads)) {
        return NULL;
    }

    std::vector< std::vector< double > > arr(nmax, std::vector< double >(nmax, 0.0));
    initdat(arr, nmax, 2*M_PI);

    PyObject* py_arr = PyList_New(nmax);
    
    for (int i = 0; i < nmax; i++) {
        PyObject* row = PyList_New(nmax);
        for (int j = 0; j < nmax; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(arr[i][j]));
        }
        PyList_SetItem(py_arr, i, row);
    }

    return Py_BuildValue("O", py_arr);
}

static PyObject* py_one_energy(PyObject* self, PyObject* args)
{
    // double one_energy(const std::vector< std::vector< double > > &arr , int ix, int iy, int nmax) 
    PyObject* py_arr;
    int ix;
    int iy;
    int nmax;
    int nthreads;

    if (!PyArg_ParseTuple(args, "Oiiii", &py_arr, &ix, &iy, &nmax, &nthreads)) {
        return NULL;
    }

    // Check if py_arr is a list
    if (!PyList_Check(py_arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    // Convert py_arr to a C++ vector of vectors of doubles
    std::vector< std::vector< double > > arr(nmax, std::vector< double >(nmax, 0.0));
    for (int i = 0; i < nmax; ++i) {
        PyObject* row = PyList_GetItem(py_arr, i);
        // Check if row is a list
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
            return NULL;
        }

        for (int j = 0; j < nmax; ++j) {
            PyObject* item = PyList_GetItem(row, j);
            // Check if item is a float
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Expected a list of lists of floats");
                return NULL;
            }
            // Populate the C++ vector
            arr[i][j] = PyFloat_AsDouble(item);
        }
    }

    // Return the using cpp implementation
    omp_set_num_threads(nthreads);
    double result = one_energy(arr, ix, iy, nmax);

    // Return double 
    return Py_BuildValue("d", result);
}

static PyObject* py_all_energy(PyObject* self, PyObject* args)
{
    // double all_energy(const std::vector< std::vector< double > > &arr, int nmax) 
    PyObject* py_arr;
    int nmax;
    int nthreads;

    // Check argument types
    if (!PyArg_ParseTuple(args, "Oii", &py_arr, &nmax, &nthreads)) {
        return NULL;
    }

    // Check arr is a list of lists 
    if (!PyList_Check(py_arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    // Convert py_arr to a C++ vector of vectors of doubles
    std::vector< std::vector< double > > arr(nmax, std::vector< double >(nmax, 0.0));
    for (int i = 0; i < nmax; ++i) {
        PyObject* row = PyList_GetItem(py_arr, i);
        // Check if row is a list
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
            return NULL;
        }

        for (int j = 0; j < nmax; ++j) {
            PyObject* item = PyList_GetItem(row, j);
            // Check if item is a float
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Expected a list of lists of floats");
                return NULL;
            }
            // Populate the C++ vector
            arr[i][j] = PyFloat_AsDouble(item);
        }
    }

    // Return the using cpp implementation
    omp_set_num_threads(nthreads);
    double result = all_energy(arr, nmax);

    // Return double
    return Py_BuildValue("d", result);
}

static PyObject* py_get_order(PyObject* self, PyObject* args)
{
    // double get_order(const std::vector< std::vector< double > > &arr, int nmax) 
    PyObject* py_arr;
    int nmax;
    int nthreads;

    if (!PyArg_ParseTuple(args, "Oii", &py_arr, &nmax, &nthreads)) {
        return NULL;
    }

    if (!PyList_Check(py_arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    // Convert py_arr to C++ vector of vectors of doubles
    std::vector< std::vector< double > > arr(nmax, std::vector< double >(nmax, 0.0));
    for (int i = 0; i < nmax; i ++) {
        PyObject* row = PyList_GetItem(py_arr, i);
        // Check if row is a list
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
            return NULL;
        }

        for (int j = 0; j < nmax; j++) {
            PyObject* item = PyList_GetItem(row, j);
            // Check if item is a float
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Expected a list of floats");
                return NULL;
            }
            // Populate the C++ vector
            arr[i][j] = PyFloat_AsDouble(item);
        }
    }
    // Return using the cpp implementation
    omp_set_num_threads(nthreads);
    double result = get_order(arr, nmax);

    return Py_BuildValue("d", result);
}

static PyObject* py_MC_step(PyObject* self, PyObject* args)
{
    // double MC_step(std::vector< std::vector< double > > &arr, double Ts, int nmax) 
    PyObject* py_arr;
    double Ts;
    int nmax;
    int nthreads;

    if (!PyArg_ParseTuple(args, "Odii", &py_arr, &Ts, &nmax, &nthreads)) {
        return NULL;
    }

    if (!PyList_Check(py_arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    // Convert py_arr to a c++ vector of vectors of doubles 
    std::vector< std::vector< double > > arr(nmax, std::vector< double >(nmax, 0.0));
    if (!py_arr_to_vector(py_arr, arr, nmax)) {
        return NULL;
    }

    omp_set_num_threads(nthreads);
    double result = MC_step(arr, Ts, nmax);

    // update the py_arr
    for (int i = 0; i < nmax; i++) {
        PyObject* row = PyList_GetItem(py_arr, i);
        for (int j = 0; j < nmax; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(arr[i][j]));
        }
    }

    return Py_BuildValue("d", result);
}

static PyMethodDef LebwohlLasherMethods[] = {
    {"LebwohlLasher", py_LebwohlLasher, METH_VARARGS, "Run the Lebwohl-Lasher simulation: <nsteps> <size> <tempurature> <threadcount>>"},
    {"energies", py_energies, METH_VARARGS, "Calculate the energy of a single cell: <lattice> <size>"},
    {"initdat", py_initdat, METH_VARARGS, "Iniitalise the lattice: <nmax>"},
    {"one_energy", py_one_energy, METH_VARARGS, "Calculate the energy of a single cell: <lattice> <ix> <iy> <size>"},
    {"all_energy", py_all_energy, METH_VARARGS, "Calculate the energy of the entire lattice: <lattice> <size>"},
    {"get_order", py_get_order, METH_VARARGS, "Calculate the order of the lattice: <lattice> <size>"},
    {"MC_step", py_MC_step, METH_VARARGS, "Perform a Monte Carlo step: <lattice> <tempurature> <size>"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef LebwohlLasherModule = {
    PyModuleDef_HEAD_INIT,
    "LebwohlLasher",
    NULL,
    -1,
    LebwohlLasherMethods
};

PyMODINIT_FUNC PyInit_LebwohlLasher(void) 
{
    return PyModule_Create(&LebwohlLasherModule);
}

}