import numpy as np
import scipy.optimize
import multiprocessing as mp
import kliff
from kliff import parallel
from kliff.pytorch_neuralnetwork import FingerprintsDataset
from kliff.pytorch_neuralnetwork import FingerprintsDataLoader

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

logger = kliff.logger.get_logger(__name__)


def energy_forces_residual(identifier, natoms, prediction, reference, data):
    """
    Parameters
    ----------
    identifier: str
      identifer of the configuration, i.e. path to the file

    natoms: int
      number of atoms in the configuration

    prediction: 1D array
      prediction computed by calculator

    reference: 1D array
      references data for the prediction

    data: dict
      user provided callback data
      aviailable keys:

      energy_weight: float (default: 1)

      forces_weight: float (default: 1)

      normalize_by_number_of_atoms: bool (default: False)
        Whether to normalize the residual by the number of atoms


    Return
    ------

    residual: 1D array


    The length of `prediction` and `reference` (call it `S`) are the same, and it
    depends on `use_energy` and `use_forces` in KIMCalculator. Assume the
    configuration contains of `N` atoms.

    1) If use_energy == True and use_forces == False
      S = 1
    `prediction[0]` is the potential energy computed by the calculator, and
    `reference[0]` is the reference energy.

    2) If use_energy == False and use_forces == True
      S = 3N
    `prediction[3*i+0]`, `prediction[3*i+1]`, and `prediction[3*i+2]` are the
    x, y, and z component of the forces on atom i in the configuration, respecrively.
    Correspondingly, `reference` is the 3N concatenated reference forces.


    3) If use_energy == True and use_forces == True
      S = 3N + 1

    `prediction[0]` is the potential energy computed by the calculator, and
    `reference[0]` is the reference energy.
    `prediction[3*i+1]`, `prediction[3*i+2]`, and `prediction[3*i+3]` are the
    x, y, and z component of the forces on atom i in the configuration, respecrively.
    Correspondingly, `reference` is the 3N concatenated reference forces.
    """

    if len(prediction) != 3*natoms + 1:
        raise ValueError(
            "len(prediction) != 3N+1, where N is the number of atoms.")

    try:
        energy_weight = data['energy_weight']
    except KeyError:
        energy_weight = 1
    try:
        forces_weight = data['forces_weight']
    except KeyError:
        forces_weight = 1
    try:
        do_normalize = data['normalize_by_number_of_atoms']
        if do_normalize:
            normalizer = natoms
        else:
            normalizer = 1
    except KeyError:
        normalizer = 1

    # TODO mornalizer
    #residual = np.subtract(prediction, reference)
    #residual[0] *= np.sqrt(energy_weight)
    #residual[1:] *= np.sqrt(forces_weight)

    residual = prediction - reference
    residual[0] *= np.sqrt(energy_weight)
    residual[1:] *= np.sqrt(forces_weight)

    return residual


def forces_residual(conf_id, natoms, prediction, reference, data):
    if data is not None:
        data['energy_weight'] = 0
    else:
        data = {'energy_weight': 0}
    return energy_forces_residual(conf_id, natoms, prediction, reference, data)


def energy_residual(conf_id, natoms, prediction, reference, data):
    if data is not None:
        data['forces_weight'] = 0
    else:
        data = {'forces_weight': 0}
    return energy_forces_residual(conf_id, natoms, prediction, reference, data)


class Loss(object):
    """Objective function class to conduct the optimization."""

    def __new__(self, calculator, nprocs=1, residual_fn=energy_forces_residual,
                residual_data=None):
        """
        Parameters
        ----------

        calculator: Calculator object

        nprocs: int
          Number of processors to parallel to run the predictors.

        residual_fn: function
          function to compute residual, see `energy_forces_residual` for an example

        residual_data: dict
          data passed to residual function

        """

        calc_type = calculator.__class__.__name__

        if calc_type == 'PytorchANNCalculator':
            return LossNeuralNetworkModel(calculator, nprocs, residual_fn, residual_data)
        else:
            return LossPhysicsMotivatedModel(calculator, nprocs, residual_fn, residual_data)


class LossPhysicsMotivatedModel(object):
    """Objective function class to conduct the optimization for PM models."""

    scipy_minimize_methods = [
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP',
        'trust-constr',
        'dogleg',
        'trust-ncg',
        'trust-exact',
        'trust-krylov']

    scipy_minimize_methods_not_supported_arguments = ['bounds']

    scipy_least_squares_methods = ['trf', 'dogbox', 'lm']

    scipy_least_squares_methods_not_supported_arguments = ['bounds']

    def __init__(self, calculator, nprocs=1, residual_fn=energy_forces_residual,
                 residual_data=None):
        """
        Parameters
        ----------

        calculator: Calculator object

        nprocs: int
          Number of processors to parallel to run the predictors.

        residual_fn: function
          function to compute residual, see `energy_forces_residual` for an example

        residual_data: dict
          data passed to residual function

        """
        if not torch_available:
            raise ImportError(
                'Please install "PyTorch" first. See: https://pytorch.org')

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data if residual_data is not None else dict()
        self.calculator_type = calculator.__class__.__name__

        self.calculator.update_model_params()

        if self.calculator_type == 'WrapperCalculator':
            calculators = self.calculator.calculators
        else:
            calculators = [self.calculator]
        for calc in calculators:
            infl_dist = calc.get_influence_distance()
            cas = calc.get_compute_arguments()
            # TODO can be parallelized
            for ca in cas:
                ca.refresh(infl_dist)

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))
#
#  def set_nprocs(self, nprocs):
#    """ Set the number of processors to be used."""
#    self.nprocs = nprocs
#
#  def set_residual_fn_and_data(self, fn, data):
#    """ Set residual function and data. """
#    self.residual_fn = fn
#    self.residual_fn_data = data

    def minimize(self, method, **kwargs):
        """ Minimize the loss.

        method: str
            minimization methods as specified at:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        kwargs: extra keyword arguments that can be used by the scipy optimizer.
        """

        if method in self.scipy_least_squares_methods:
            for i in self.scipy_least_squares_methods_not_supported_arguments:
                if i in kwargs:
                    raise LossError('Argument "{}" should not be set through the '
                                    '"minimize" method.'.format(i))
            bounds = self.calculator.get_opt_params_bounds()
            kwargs['bounds'] = bounds
            logger.info('Start minimization using scipy least squares method: {}.'
                        .format(method))
            result = self._scipy_optimize_least_squares(method, **kwargs)
            logger.info('Finish minimization using scipy least squares method: {}.'
                        .format(method))

        elif method in self.scipy_minimize_methods:
            for i in self.scipy_minimize_methods_not_supported_arguments:
                if i in kwargs:
                    raise LossError('Argument "{}" should not be set through the '
                                    '"minimize" method.'.format(i))
            bounds = self.calculator.get_opt_params_bounds()
            for i in range(len(bounds)):
                lb = bounds[i][0]
                ub = bounds[i][1]
                if lb is None:
                    bounds[i][0] = -np.inf
                if ub is None:
                    bounds[i][1] = np.inf
            kwargs['bounds'] = bounds
            logger.info('Start minimization using scipy optimize method: {}.'
                        .format(method))
            result = self._scipy_optimize_minimize(method, **kwargs)
            logger.info('Finish minimization using scipy optimize method: {}.'
                        .format(method))

        else:
            raise Exception('minimization method "{}" not supported.'.format(method))

        # update final optimized paramters
        self.calculator.update_params(result.x)
        return result

    def get_residual(self, x):
        """ Compute the residual.

        This is a callable for optimizing method in scipy.optimize.least_squares,
        which is passed as the first positional argument.

        Parameters
        ----------

        x: 1D array
          optimizing parameter values
        """

        # publish params x to predictor
        self.calculator.update_params(x)

        cas = self.calculator.get_compute_arguments()

        if self.calculator_type == 'WrapperCalculator':
            calc_list = self.calculator.get_calculator_list()
            X = zip(cas, calc_list)
            if self.nprocs > 1:
                residuals = parallel.parmap3(
                    self._get_residual_single_config,
                    X,
                    self.nprocs,
                    self.residual_fn,
                    self.residual_data)
                residual = np.concatenate(residuals)
            else:
                residual = []
                for ca, calc in X:
                    current_residual = self._get_residual_single_config(
                        ca,
                        calc,
                        self.residual_fn,
                        self.residual_data)
                    residual = np.concatenate((residual, current_residual))

        else:
            if self.nprocs > 1:
                residuals = parallel.parmap2(
                    self._get_residual_single_config,
                    cas,
                    self.nprocs,
                    self.calculator,
                    self.residual_fn,
                    self.residual_data)
                residual = np.concatenate(residuals)
            else:
                residual = []
                for ca in cas:
                    current_residual = self._get_residual_single_config(
                        ca,
                        self.calculator,
                        self.residual_fn,
                        self.residual_data)
                    residual = np.concatenate((residual, current_residual))

        return residual

    def get_loss(self, x):
        """ Compute the loss.

        This is a callable for optimizing method in scipy.optimize,minimize,
        which is passed as the first positional argument.

        Parameters
        ----------

        x: 1D array
          optimizing parameter values
        """
        residual = self.get_residual(x)
        loss = 0.5*np.linalg.norm(residual)**2
        return loss

    def _scipy_optimize_least_squares(self, method, **kwargs):
        residual = self.get_residual
        x = self.calculator.get_opt_params()
        return scipy.optimize.least_squares(residual, x, method=method, **kwargs)

    def _scipy_optimize_minimize(self, method, **kwargs):
        loss = self.get_loss
        x = self.calculator.get_opt_params()
        return scipy.optimize.minimize(loss, x, method=method, **kwargs)

    def _get_residual_single_config(self, ca, calculator, residual_fn, residual_data):

        # prediction data
        calculator.compute(ca)
        pred = calculator.get_prediction(ca)

        # reference data
        ref = calculator.get_reference(ca)

        conf = ca.conf
        identifier = conf.get_identifier()
        natoms = conf.get_number_of_atoms()

        residual = residual_fn(identifier, natoms, pred, ref, residual_data)

        return residual

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, trackback):
        # if there is expections, raise it (not for KeyboardInterrupt)
        if exec_type is not None and exec_type is not KeyboardInterrupt:
            return False  # return False will cause Python to re-raise the expection


class LossNeuralNetworkModel(object):
    """Objective function class to conduct the optimization for ML models."""

    torch_minimize_methods = [
        'Adadelta',
        'Adagrad',
        'Adam',
        'SparseAdam',
        'Adamax',
        'ASGD',
        'LBFGS',
        'RMSprop',
        'Rprop',
        'SGD']

    def __init__(self, calculator, nprocs=1, residual_fn=energy_forces_residual,
                 residual_data=None):
        """
        Parameters
        ----------

        calculator: Calculator object

        nprocs: int
          Number of processors to parallel to run the predictors.

        residual_fn: function
          Function to compute residual, see `energy_forces_residual` for an example.

        residual_data: dict
          Data passed to residual function.

        """

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_fn = residual_fn
        self.residual_data = residual_data if residual_data is not None else dict()

        self.calculator_type = calculator.__class__.__name__

        logger.info('"{}" instantiated.'.format(self.__class__.__name__))
#
#  def set_nprocs(self, nprocs):
#    """ Set the number of processors to be used."""
#    self.nprocs = nprocs
#
#  def set_residual_fn_and_data(self, fn, data):
#    """ Set residual function and data. """
#    self.residual_fn = fn
#    self.residual_fn_data = data

    def minimize(self, method, batch_size=100, num_epochs=1000, **kwargs):
        """ Minimize the loss.

        method: str
            PyTorch optimization methods, and aviliable ones are:
            [`Adadelta`, `Adagrad`, `Adam`, `SparseAdam`, `Adamax`, `ASGD`,
             `LBFGS`, `RMSprop`, `Rprop`, `SGD`]
            For also: https://pytorch.org/docs/stable/optim.html

        kwargs: extra keyword arguments that can be used by the PyTorch optimizer.
        """

        self.method = method
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # data loader
        fname = self.calculator.get_train_fingerprints_path()
        fp = FingerprintsDataset(fname)
        self.data_loader = FingerprintsDataLoader(dataset=fp, num_epochs=num_epochs)

        # optimizing
        optimizer = getattr(torch.optim, method)(
            self.calculator.model.parameters(), **kwargs)
        n = 0
        epoch = 0
        DATASET_SIZE = len(self.calculator.configs)
        while True:
            try:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                loss = self.get_loss()
                loss.backward()
                optimizer.step()
                epoch_new = n*batch_size // DATASET_SIZE
                if epoch_new > epoch:
                    epoch = epoch_new
                    print('Epoch = {}, loss = {}'.format(epoch, loss))
                n += 1
            except StopIteration:
                break

    def get_loss(self):
        """
        """
        loss = 0
        for _ in range(self.batch_size):
            # raise StopIteration error if out of bounds; This will ignore the last
            # chunk of data whose size is smaller than `batch_size`
            inp = self.data_loader.next_element()
            residual = self._get_residual_single_config(
                inp, self.calculator, self.residual_fn, self.residual_data)
            c = torch.sum(torch.pow(residual, 2))
            loss += c
        # TODO  maybe divide batch_size elsewhere
        loss /= self.batch_size
        return loss

    def _get_residual_single_config(self, inp, calculator, residual_fn, residual_data):

        # prediction data
        results = calculator.compute(inp)
        pred_energy = results['energy']
        pred_forces = results['forces']

        if calculator.use_energy:
            pred = torch.tensor([pred_energy])
            ref = torch.tensor([inp['energy'][0]])
        if calculator.use_forces:
            ref_forces = inp['forces'][0]
            if calculator.use_energy:
                pred = torch.cat((pred, pred_forces.reshape((-1,))))
                ref = torch.cat((ref, ref_forces.reshape((-1,))))
            else:
                pred = pred_forces.reshape((-1,))
                ref = ref_forces.reshape((-1,))

        identifier = inp['name'][0]
        species = inp['species'][0]
        natoms = len(species)

        residual = residual_fn(identifier, natoms, pred, ref, residual_data)

        return residual


class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg

    def __expr__(self):
        return self.msg
