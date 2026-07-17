import unittest

from canonicity.generation import (
    GenerationSettings,
    PromptContext,
    _eos_token_ids,
    _hardware_signature,
    _load_model_exact,
    _resolve_repo_commit,
    _resolve_device,
    _resolve_dtype,
    generate_samples,
)


class _UnavailableCuda:
    @staticmethod
    def is_available():
        return False


class _ReportedMps:
    @staticmethod
    def is_available():
        return True


class DeviceResolutionTests(unittest.TestCase):
    def test_auto_falls_back_when_mps_cannot_allocate(self):
        class Torch:
            cuda = _UnavailableCuda()
            backends = type("Backends", (), {"mps": _ReportedMps()})()

            @staticmethod
            def empty(*args, **kwargs):
                raise RuntimeError("backend is reported but unusable")

        self.assertEqual(_resolve_device(Torch(), "auto"), "cpu")

    def test_explicit_device_is_not_overridden(self):
        self.assertEqual(_resolve_device(object(), "mps"), "mps")

    def test_auto_dtype_preserves_checkpoint_precision(self):
        self.assertEqual(_resolve_dtype(object(), "auto", "cuda"), "auto")

    def test_explicit_dtype_requests_a_cast(self):
        torch = type("Torch", (), {"float16": "fp16"})()
        self.assertEqual(_resolve_dtype(torch, "float16", "cpu"), "fp16")

    def test_only_configured_eos_tokens_terminate_generation(self):
        model = type(
            "Model",
            (),
            {
                "generation_config": type(
                    "GenerationConfig", (), {"eos_token_id": [2, 3]}
                )(),
                "config": type("Config", (), {"eos_token_id": 4})(),
            },
        )()
        tokenizer = type(
            "Tokenizer", (), {"eos_token_id": 5, "all_special_ids": [1, 2, 3, 4, 5]}
        )()

        self.assertEqual(_eos_token_ids(model, tokenizer), {2, 3})

    def test_native_composite_config_is_passed_through_unchanged(self):
        composite = type("CompositeConfig", (), {"text_config": object()})()
        seen = {}

        class Loader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                seen.update({"model_id": model_id, **kwargs})
                return "model", {}

        model = _load_model_exact(Loader, "example/model", composite, {"dtype": "auto"})

        self.assertEqual(model, "model")
        self.assertIs(seen["config"], composite)
        self.assertTrue(seen["output_loading_info"])

    def test_incomplete_weight_load_is_rejected(self):
        class Loader:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                return object(), {"unexpected_keys": ["language_model.weight"]}

        with self.assertRaises(RuntimeError):
            _load_model_exact(Loader, "example/model", object(), {})

    def test_full_commit_does_not_require_a_network_resolution(self):
        class Api:
            def model_info(self, *args, **kwargs):
                raise AssertionError("already immutable")

        commit = "a" * 40
        self.assertEqual(_resolve_repo_commit(Api(), "example/model", commit), commit)

    def test_branch_is_resolved_to_one_immutable_commit(self):
        class Api:
            def model_info(self, repo_id, revision):
                self.arguments = (repo_id, revision)
                return type("Info", (), {"sha": "b" * 40})()

        api = Api()
        self.assertEqual(
            _resolve_repo_commit(api, "example/model", "main"),
            "b" * 40,
        )
        self.assertEqual(api.arguments, ("example/model", "main"))

    def test_cpu_hardware_signature_ignores_unused_cuda_devices(self):
        class Cuda:
            @staticmethod
            def is_available():
                return True

        torch = type("Torch", (), {"cuda": Cuda()})()

        signature = _hardware_signature(torch, "cpu")

        self.assertEqual(signature["resolved_device"], "cpu")
        self.assertNotIn("cuda_devices", signature)

    def test_programmatic_generation_rejects_two_placement_modes(self):
        settings = GenerationSettings(
            model_id="example/model",
            tokenizer_id=None,
            revision=None,
            samples_per_context=1,
            max_new_tokens=1,
            batch_size=1,
            seed=0,
            device="cuda:1",
            dtype="auto",
            device_map="auto",
        )

        with self.assertRaises(ValueError):
            generate_samples(settings, (PromptContext("unconditional", None),))


if __name__ == "__main__":
    unittest.main()
