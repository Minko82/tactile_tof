from __future__ import annotations

import inspect

from sim.mechanics.gpu_stepper import GpuFrameStepper
from sim.mechanics.interactive_runner import InteractiveTouchController
from sim.mechanics.newton_runner import TouchMechanicsControllerV2


def test_physics_substep_builder_has_no_host_array_readback():
    source = inspect.getsource(GpuFrameStepper._build_frame_operations)
    assert ".numpy(" not in source
    assert "_contact_summary" not in source
    assert "evaluate_candidate" in source
    assert "_conditionally_accept_particles" in source


def test_both_runners_delegate_physics_to_shared_gpu_stepper():
    scripted = inspect.getsource(TouchMechanicsControllerV2.simulate)
    interactive = inspect.getsource(InteractiveTouchController.simulate)
    assert "gpu_stepper.run_frame" in scripted
    assert "gpu_stepper.run_frame" in interactive
    assert "_contact_summary" not in scripted
    assert "_contact_summary" not in interactive


def test_picker_readback_occurs_once_in_rendered_frame_ingest_only():
    consume = inspect.getsource(InteractiveTouchController._consume_picker_target)
    synchronize = inspect.getsource(InteractiveTouchController._sync_picker_point)
    drag = inspect.getsource(InteractiveTouchController._on_mouse_drag)
    assert consume.count("pick_body.numpy()") == 1
    assert consume.count("pick_state.numpy()") == 1
    assert ".numpy()" not in synchronize
    assert ".numpy()" not in drag
