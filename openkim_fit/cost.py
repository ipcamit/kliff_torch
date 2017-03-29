from __future__ import print_function, division
import numpy as np
import multiprocessing as mp
import collections
from modelparams import ModelParams
from error import InputError

class Cost:
    """Objective cost function that will be minimized.

    Parameters
    ----------

    params, ModelParams object
        It's method 'update_params' will be used to update parameters from
        the optimizer.

    nprocs, int
        Number of processors to parallel to run the predictors.
    verbose, int
        Integer code to denote to whether echo running cost to screen.
        0, do not echo anything
        1, echo running cost
    """
    def __init__(self, params, nprocs=1, verbose=0):

        self.params = params
        self.nprocs = nprocs
        self.verbose = verbose
        self.pred_obj = []
        self.ref = []
        self.weight = []
        self.func = []
        self.pred_obj_group = None
        self.ref_group = None
        self.weight_group = None
        self.func_group = None

    def __enter__(self):
      return self

    def add(self, pred_obj, reference, weight=1, func=None):
        """Add a contribution part to the cost.

        Parameters
        ----------

        pred_obj: predictor object.
            It must have the 'get_prediction()' method, which returns a float
            list that has the same length as "reference". Optionally, it can have
            the 'update_params()' method that updates the parameters from the
            ModelParams class. However, for OpenKIM test, this is impossible, since
            we have no access to its KIM object. The way we handle it is to crack
            the KIM api to read parameters at Model initialization stage.

        reference: float list
            reference data

        weight: float or list
            weight for the prediction and reference data. If a float number
            is entered, all the prediction and reference data entries will
            use the same value. Its length should match reference data if
            "func" is not set, otherwise, its length should match the length
            of the return value of "func".

        func(pred_obj.get_prediction(), reference):
            a function to generate residual using a method rather than the
            default one. It takes two R^m list arguments, and returns a R^n
            list.
        """

        # check correct length
        pred = pred_obj.get_prediction()
        len_pred = len(pred)
        if len_pred != len(reference):
            raise InputError('Cost.add(): lengths of prediction and reference data '
                             'do not match.')
        # construct weight list according to input type
        if func is None:
            len_resid = len_pred
        else:
            residual = func(pred,reference)
            if not isinstance(residual, (collections.Sequence, np.ndarray)):
                raise InputError('Cost.add(): return value of "{}" should be a list.'.format(func.__name__))
            len_resid = len(residual)

        if isinstance(weight, (collections.Sequence, np.ndarray)):
            if len(weight) != len_resid:
                raise InputError('Cost.add(): length of return value of "{}" '
                                 'and that of weight do not match.'.format(func.__name__))
        else:
            weight = [weight for i in range(len_resid)]

        # record data
        self.pred_obj.append(pred_obj)
        self.ref.append(reference)
        self.weight.append(weight)
        self.func.append(func)
        # group objects added so far
        self._group_preds()


    def _group_preds(self):
        """ Group the predictors (and the associated references, weights and wrapper
        functions) into 'nprocs' groups, so that each group can be processed by a
        processor. The predictors are randomly assigned to each group to account for
        load balance in each group.
        """
#NOTE we may not want the shuffling while developing the code, such that it gives
# deterministic results
        # shuffle before grouping, in case some processors get quite heavy jobs but
        # others get light ones.
#        combined = zip(self.pred_obj, self.ref, self.weight, self.func)
#        np.random.shuffle(combined)
#        self.pred_obj[:], self.ref[:], self.weight[:], self.func[:] = zip(*combined)

        # grouping
        self.pred_obj_group= np.array_split(self.pred_obj, self.nprocs)
        self.ref_group= np.array_split(self.ref, self.nprocs)
        self.weight_group= np.array_split(self.weight, self.nprocs)
        self.func_group= np.array_split(self.func, self.nprocs)


    def get_residual(self, x0):
        """ Compute the residual for the cost.
        This is a callable for optimizer that usually passed as the first positional
        argument.

        Parameters
        ----------

        x0: list
            optimizing parameter values
        """
        # update params from minimizer to ModelParams
        self._update_params(x0)


        # create jobs and start
        jobs = []
        pipe_list = []
        for g in range(self.nprocs):
            # create pipes
            recv_end, send_end = mp.Pipe(False)
            # get data for this group
            pred = self.pred_obj_group[g]
            ref = self.ref_group[g]
            weight = self.weight_group[g]
            func = self.func_group[g]
            p = mp.Process(target=self._get_residual_group, args=(pred,ref,weight,func,send_end,))
            p.start()
            jobs.append(p)
            pipe_list.append(recv_end)

        # we should place recv() before join()
        residuals = [x.recv() for x in pipe_list]
        # wait for the all the jobs to complete
        for p in jobs:
            p.join()

        residuals = np.concatenate(residuals)

        if self.verbose == 1:
            cost = 0.5*np.linalg.norm(residuals)**2
            print ('cost: {:18.10e}'.format(cost))

        return residuals


    def get_cost(self, x0):
        """ Compute the cost.
        This can be used as the callable for optimizer (e.g. scipy.optimize.minimize
        method='CG')

        Parameters
        ----------

        x0: list
            optimizing parameter values
        """
        residual = self.get_residual(x0)
        cost = 0.5*np.linalg.norm(residual)**2
        return cost


    def _update_params(self, x0):
        """Update parameter values from minimizer to ModelParams object.

        Parameters
        ----------

        x0: list
            optimizing parameter values
        """
        self.params.update_params(x0)

# NOTE (we we support print to screen every N steps)
# print to file while minimizion
        self.params.echo_params('params.txt')

    def _get_residual_group(self, pred_g, ref_g, weight_g, func_g, send_end):
        """Helper function to do the computation of residuals.
        """
        residual = []
        for i in range(len(pred_g)):
            pred = pred_g[i].get_prediction()
            ref = ref_g[i]
            weight = np.sqrt(weight_g[i])
            func = func_g[i]
            if func is None:
                difference = np.subtract(pred, ref)
            else:
                difference = func(pred, ref)
            tmp_resid = np.multiply(weight, difference)
            residual = np.concatenate((residual, tmp_resid))
        send_end.send(residual)


    def _error_report(self, fname='ERROR_REPORT'):
      """Write error of each configuration to fname.
      """
      residuals = self.get_residual(self.params.get_x0())
      cost_all = 0.5*residuals**2

      # the order of residual is the same as grouped prediction objects
      pred_objs = np.concatenate(self.pred_obj_group)
      refs = np.concatenate(self.ref_group)
      weights = np.concatenate(self.weight_group)

      start = 0
      with open (fname, 'w') as fout:
          fout.write('Total error: {:18.10e}\n\n'.format(sum(cost_all)))
          fout.write('Below is error by each configuration.\n')
          for p_obj, r, w in zip(pred_objs, refs, weights):
              identifier = p_obj.conf.id
              p = p_obj.get_prediction()
              end = start + len(p)
              cost_config = sum(cost_all[start:end])
              start = end
              fout.write('\nconfig: {},    config error:{:18.10e}\n'.format(identifier, cost_config))
              fout.write('     Prediction      Reference     Difference     Diff./Ref.       Weight         Error\n')
              for i,j,k in zip(p,r,w):
                  diff = i-j
                  try:
                    ratio = float(diff)/float(j) # change np.float to float to catch error
                  except ZeroDivisionError:
                    ratio = np.inf
                  error = 0.5*k*diff**2
                  fout.write('  {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e} {:14.6e}\n'.format(i,j,diff,ratio,k,error))


    def __exit__(self, exec_type, exec_value, trackback):

        # if there is expections, raise it (not for KeyboardInterrupt)
        if exec_type is not None and exec_type is not KeyboardInterrupt:
            return False # return False will cause Python to re-raise the expection

        # write error report
        self._error_report()

        #write fitted params to `FINAL_FITTED_PARAMS' and stdout at end.
        self.params.echo_params(fname='FINAL_FITTED_PARAMS')
        self.params.echo_params()
