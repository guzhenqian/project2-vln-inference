"""Prepare/freeze the six precision extension kernels before native startup."""
from navigation_repair_runtime import NavigationRepairModel, WB, AB, EXTENSION


class PrecisionRepairModel(NavigationRepairModel):
    def _precision_evidence(self):
        from navigation_precision_repair_kernels import ACTIVATION_RECIPES
        return dict(super()._precision_evidence(), activation_recipe=ACTIVATION_RECIPES[AB],
                    kernel_variant='balanced', launch_policy='startup prepared and frozen')

    def forward(self, *args, **kwargs):
        if not self._fix_execution_verified:
            assert EXTENSION and self._awq_loaded_weights == self._awq_expected_weights
            from navigation_precision_repair_kernels import prepare_projection, freeze
            for module in self._awq_objects.values():
                prepare_projection(module.qweight, module.scales, module.qzeros, WB, AB)
            self._repair_kernel_table = freeze()
        return super().forward(*args, **kwargs)
