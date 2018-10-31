from itertools import chain
from hashlib import sha256
from zlib import adler32 as hash_func
import numpy as np
from scipy.sparse import spdiags, coo_matrix, dia_matrix
import cvxpy
from cvxpy.error import SolverError


params = {
    "alpha_derv_2": 120.0,
    "solver": {
        "name": "ECOS",
        "verbose": False,
        "max_iters": 3505,
        "tolerances": {
            "abstol": 1e-06,
            "reltol": 1e-07,
            "feastol": 1e-06,
            "abstol_inacc": 0.001,
            "reltol_inacc": 0.0001,
            "feastol_inacc": 0.0001
        }
    }
}


def hash_string_obj(string):
    x_bytes = string.encode()
    return sha256(x_bytes)


def hash_string(string, n_chars=16):
    hash_object = hash_string_obj(string)
    hash_val = hash_object.hexdigest()
    return hash_val[0:n_chars]


def describe_data(data):
    print(data.__class__)
    print(len(data))
    print(data[0].keys())
    for k, v in data[0].items():
        print(k, v.__class__)


def check_data(problem):
    solver = params['solver']['name']
    data = problem.get_problem_data(solver)
    # describe_data(data)
    # for ECOS ONLY
    variables = data[0].keys()
    hashes = []
    for var in variables:
        if var == 'dims':
            continue

        arr = data[0][var]
        if arr is None:
            continue

        if not isinstance(arr, np.ndarray):
            arr = arr.todense()

        arr = arr.copy(order='C')
        arr.flags.writeable = False
        hash_val = hash_func(arr)
        print('Hash for %s: %s' % (var, hash_val))
        if var == 'c':
            print(arr)

        hashes.append(hash_val)

    hash_all = hash_string(hashes.__repr__())
    return hash_all


def second_derivative_matrix_nes(x, a_min=0.0, a_max=None, scale_free=False):
    """
    Get the second derivative matrix for non-equally spaced points
    :param : x numpy array of x-values
    :param : a_min, float. Allows for modification where the x-values
             can't get any closer than this. (Not technically the sec derv)
    :param : a_max, float. Same but max.
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivative of x
    assumes points are sorted
    """
    n = len(x)
    m = n - 2

    values = []
    for i in range(1, n-1):
        # These are all positive if sorted
        a0 = float(x[i+1] - x[i])
        a2 = float(x[i] - x[i-1])

        assert (a0 >= 0) and (a2 >= 0), "Points do not appear to be sorted"

        # And neither cab be zero
        assert (a0 > 0) and (a2 > 0), "Second derivative doesn't exist for repeated points"

        # Now allow for a min and max on the differences
        # of the x separations
        if a_max is not None:
            a0 = min(a0, a_max)
            a2 = min(a2, a_max)

        a0 = max(a0, a_min)
        a2 = max(a2, a_min)
        a1 = a0 + a2

        if scale_free:
            # Just subtract derivatives
            # don't divide again by the length scale
            scf = a1/2.0
        else:
            scf = 1.0

        vals = [2.0*scf/(a1*a2), -2.0*scf/(a0*a2), 2.0*scf/(a0*a1)]
        values.extend(vals)

    i = list(chain(*[[_] * 3 for _ in range(m)]))
    j = list(chain(*[[_, _ + 1, _ + 2] for _ in range(m)]))

    d2 = coo_matrix((values, (i, j)), shape=(m, n))

    return dia_matrix(d2)


def get_solver(solver_name):
    solvers = {'CVXOPT': cvxpy.CVXOPT,
               'ECOS': cvxpy.ECOS,
               'SCS': cvxpy.SCS}

    return solvers[solver_name]


def hinge_norm(diff):
    return cvxpy.sum(cvxpy.maximum(diff, 0))


class TSmodelSimple(object):
    def __init__(self, time, values, use_hinge=True):
        assert (time == time[np.argsort(time)]).all()

        self.use_hinge = use_hinge
        self.solver_params = params['solver']
        self.params = params
        self.time = time
        self.values = values
        self.n_points = len(time)
        self.hash_val = None
        self.solver = get_solver(self.solver_params['name'])
        self._create_model()
        self.hash_all_data = check_data(self.problem)

    def _create_model(self):
        # Create the CVXPY model

        # Define the variables
        self.model_var = cvxpy.Variable(self.n_points)

        # terms for objective function

        d2 = second_derivative_matrix_nes(self.time, scale_free=True)

        # Objective terms

        diff = self.values - self.model_var

        # Seems NOT to fail when use_hinge is False
        # But should be equivalent

        if self.use_hinge:
            hinge_norm_right = hinge_norm(diff)
            hinge_norm_left = hinge_norm(-diff)

            self.diff_obj = hinge_norm_left + hinge_norm_right
        else:
            self.diff_obj = cvxpy.norm(diff, 1)

        self.second_deriv_obj = self.params['alpha_derv_2'] \
            * cvxpy.norm(d2 * self.model_var, 1)

        self.objective = self.diff_obj + self.second_deriv_obj

        # Minimize the objective function
        obj = cvxpy.Minimize(self.objective)

        # Make constraints

        # Finally create, the Problem to be solved
        self.problem = cvxpy.Problem(obj)

    def _fit(self, solver_name):
        tols = self.solver_params['tolerances']
        obj_min = self.problem.solve(solver=solver_name,
                                     verbose=self.solver_params['verbose'],
                                     max_iters=self.solver_params['max_iters'],
                                     abstol=tols['abstol'],
                                     reltol=tols['reltol'],
                                     feastol=tols['feastol'],
                                     abstol_inacc=tols['abstol_inacc'],
                                     reltol_inacc=tols['reltol_inacc'],
                                     feastol_inacc=tols['feastol_inacc'],
                                     warm_start=False)

        print('OBJ_MIN', obj_min)

        self.objective_value = obj_min

    def fit(self):
        self._fit(self.solver_params['name'])
        if self.problem.status != 'optimal':
            raise SolverError

        # self.print_obj()

        hash_val = hash_string(self.model_var.value.__repr__(), 20)
        self.hash_val = hash_val

    def print_obj(self):
        print('obj: ', self.objective.value)
        print('obj SS: ', self.diff_obj.value)


def test_deterministic_simple():
    time = np.array([0., 89., 90., 100.0])
    values = np.array([0., 1000., 2000., 2700.0])

    model = TSmodelSimple(time, values)
    # assert False
    model.fit()
    hash_val_orig = model.hash_val
    hash_data_orig = model.hash_all_data

    print("HASH_VAL_ORIG: %s" % hash_val_orig)
    print("HASH_DATA_ORIG: %s" % hash_data_orig)
    print('-------------------------')
    obj_orig = model.objective_value

    n = 20
    n_bad = 0
    for i in range(n):
        model = TSmodelSimple(time, values)
        model.fit()
        print("HASH_VAL_ITER_%s: %s" %(i, model.hash_val))
        print("HASH_DATA_ITER_%s: %s" % (i, model.hash_all_data))

        if model.hash_all_data != hash_data_orig:
            print('Input data has changed')
            # assert False

        obj = model.objective_value

        if obj != obj_orig:
            print('Fit model appears to be non-deterministic on iter %s' % (i+1))
            n_bad += 1

        print('-------------------------')

    if n_bad > 0:
        print('Model NOT deterministic')
        assert False

    print('OK')


def check_deterministic_input_data_only(use_hinge=True):
    # only problem setup. CVXPY still not deterministic even before
    # calling ecos solver
    time = np.array([0., 89., 90., 100.0])
    values = np.array([0., 1000., 2000., 2700.0])

    model = TSmodelSimple(time, values, use_hinge=use_hinge)
    hash_data_orig = model.hash_all_data

    print("HASH_DATA_ORIG: %s" % hash_data_orig)
    print('-------------------------')

    n = 20
    n_bad = 0
    hashes_unique = set()
    for i in range(n):
        model = TSmodelSimple(time, values, use_hinge=use_hinge)
        print("HASH_DATA_ITER_%s: %s" % (i, model.hash_all_data))

        hash_data = model.hash_all_data
        hashes_unique.add(hash_data)
        if hash_data != hash_data_orig:
            print('Input data has changed')
            n_bad += 1

        print('-------------------------')

    if n_bad > 0:
        print('Model NOT deterministic')
        print(n_bad, ' changes from original')
        print('number variants: %s' % len(hashes_unique))
        print('set of hashes', hashes_unique)
        assert False

    print('OK')


def test_deterministic_input_data_only():
    check_deterministic_input_data_only(use_hinge=True)


def test_deterministic_input_data_only_no_hinge():
    check_deterministic_input_data_only(use_hinge=False)


if __name__ == "__main__":
    test_deterministic_input_data_only_no_hinge()
    test_deterministic_input_data_only()
    test_deterministic_simple()
