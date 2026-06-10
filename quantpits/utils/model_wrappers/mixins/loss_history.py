# LossHistoryMixin — captures per-epoch train/valid loss without modifying
# the original model code.  Works by temporarily monkey-patching self.test_epoch
# during fit() so that the loss values (which the parent's fit() computes but
# discards) are intercepted and stored into evals_result.
#
# Usage (wrapper file):
#   from quantpits.utils.model_wrappers.pytorch_gru_ic import GRU as _Base
#   from quantpits.utils.model_wrappers.loss_history_mixin import LossHistoryMixin
#   class GRU(LossHistoryMixin, _Base):
#       pass


class LossHistoryMixin:
    """Mixin that captures per-epoch train/valid loss alongside scores.

    The parent model's fit() typically does::

        train_loss, train_score = self.test_epoch(train_data)  # loss discarded!
        evals_result["train"].append(train_score)

    This mixin intercepts test_epoch's return value and records the loss
    portion into side channels.  After super().fit() returns the losses are
    merged into evals_result::

        evals_result["train_loss"] = [epoch_0, epoch_1, ...]
        evals_result["valid_loss"] = [epoch_0, epoch_1, ...]
    """

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        # Side channels — the wrapped test_epoch writes here
        self._lh_train_loss = []
        self._lh_valid_loss = []
        self._lh_call_count = 0

        original_test_epoch = self.test_epoch

        def _capture(*args, **kwargs):
            result = original_test_epoch(*args, **kwargs)
            loss_val = self._lh_extract_loss(result)
            if loss_val is not None:
                if self._lh_call_count % 2 == 0:
                    self._lh_train_loss.append(loss_val)
                else:
                    self._lh_valid_loss.append(loss_val)
                self._lh_call_count += 1
            return result

        self.test_epoch = _capture
        try:
            ret = super().fit(dataset, evals_result=evals_result,
                              save_path=save_path)
        except TypeError as e:
            if "save_path" in str(e):
                ret = super().fit(dataset, evals_result=evals_result)
            else:
                raise
        finally:
            self.test_epoch = original_test_epoch

        if self._lh_train_loss:
            evals_result["train_loss"] = self._lh_train_loss
        if self._lh_valid_loss:
            evals_result["valid_loss"] = self._lh_valid_loss

        del self._lh_train_loss, self._lh_valid_loss, self._lh_call_count
        return ret

    @staticmethod
    def _lh_extract_loss(result):
        """Extract a scalar loss from test_epoch's return value.

        Handles three patterns found across models:
        - (loss, score) tuple           → result[0]         (most NN)
        - dict with 'mse'/'loss' key    → result['mse']     (ADARNN)
        - 4-tuple with metrics dict     → result[0]['MSE']  (TRA)
        """
        if isinstance(result, (tuple, list)) and len(result) >= 1:
            first = result[0]
            if isinstance(first, dict):
                return float(first.get('MSE', first.get('mse',
                       first.get('loss', None))) or 0)
            if first is not None:
                return float(first)
        elif isinstance(result, dict):
            return float(result.get('mse', result.get('loss', None)) or 0)
        return None
