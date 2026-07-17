import unittest
from importlib import metadata as importlib_metadata

from canonicity.core import SampledSequence
from canonicity.generation import (
    GenerationSettings,
    PromptContext,
    _attention_backend_provenance,
    _eos_token_ids,
    _hardware_signature,
    _load_model_exact,
    _parameter_footprint_by_device,
    _placement_summary,
    _resolve_repo_commit,
    _resolve_device,
    _resolve_dtype,
    _scan_checkpointed_batches,
    _sampling_progress_message,
    _validate_attention_applicability,
    _validate_resolved_attention,
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


class AttentionBackendTests(unittest.TestCase):
    @staticmethod
    def _torch(cuda_available=True):
        cuda = type(
            "Cuda",
            (),
            {"is_available": staticmethod(lambda: cuda_available)},
        )()
        return type("Torch", (), {"__version__": "2.8.0", "cuda": cuda})()

    @staticmethod
    def _transformers(available=True):
        utils = type(
            "Utils",
            (),
            {"is_flash_attn_2_available": staticmethod(lambda: available)},
        )()
        return type("Transformers", (), {"utils": utils})()

    def test_not_applicable_has_no_attention_provider(self):
        provenance = _attention_backend_provenance(
            object(),
            object(),
            "not_applicable",
            "cpu",
        )

        self.assertEqual(provenance["attention_provider"], "not_applicable")
        self.assertIsNone(provenance["attention_provider_version"])

    def test_sdpa_is_bound_to_the_torch_version(self):
        provenance = _attention_backend_provenance(
            self._torch(),
            object(),
            "sdpa",
            "cuda:0",
        )

        self.assertEqual(provenance["attention_provider"], "torch")
        self.assertEqual(provenance["attention_provider_version"], "2.8.0")

    def test_native_flash_attention_provider_is_recorded(self):
        provenance = _attention_backend_provenance(
            self._torch(),
            self._transformers(),
            "flash_attention_2",
            "cuda:0",
            distribution_version=lambda name: "2.8.3",
        )

        self.assertEqual(provenance["attention_provider"], "flash-attn")
        self.assertEqual(provenance["attention_provider_version"], "2.8.3")

    def test_flash_attention_never_uses_an_alternate_provider(self):
        def missing_distribution(name):
            raise importlib_metadata.PackageNotFoundError(name)

        with self.assertRaisesRegex(RuntimeError, "No alternate kernel"):
            _attention_backend_provenance(
                self._torch(),
                self._transformers(),
                "flash_attention_2",
                "cuda:0",
                distribution_version=missing_distribution,
            )

    def test_incompatible_native_flash_attention_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not compatible"):
            _attention_backend_provenance(
                self._torch(),
                self._transformers(available=True),
                "flash_attention_2",
                "cuda:0",
                distribution_version=lambda name: "2.2.0",
            )

    def test_flash_attention_rejects_cpu_inference(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            _attention_backend_provenance(
                self._torch(cuda_available=False),
                self._transformers(),
                "flash_attention_2",
                "cpu",
                distribution_version=lambda name: "2.8.3",
            )

    def test_attention_applicability_is_architectural(self):
        attention_model = type(
            "AttentionModel",
            (),
            {"_supports_attention_backend": True},
        )
        attention_free_model = type(
            "AttentionFreeModel",
            (),
            {"_supports_attention_backend": False},
        )

        _validate_attention_applicability(attention_model, "sdpa")
        _validate_attention_applicability(
            attention_free_model,
            "not_applicable",
        )
        with self.assertRaisesRegex(ValueError, "uses attention"):
            _validate_attention_applicability(
                attention_model,
                "not_applicable",
            )
        with self.assertRaisesRegex(ValueError, "attention-free"):
            _validate_attention_applicability(
                attention_free_model,
                "flash_attention_2",
            )

    def test_loaded_flash_attention_must_match_and_be_cuda_half_precision(self):
        config = type(
            "Config",
            (),
            {"_attn_implementation": "flash_attention_2"},
        )()

        class Weight:
            device = "cuda:1"
            dtype = "torch.bfloat16"

        projection = type("Projection", (), {"weight": Weight()})()
        attention = type(
            "Attention",
            (),
            {
                "config": config,
                "q_proj": projection,
                "k_proj": projection,
                "v_proj": projection,
            },
        )()
        model = type(
            "Model",
            (),
            {
                "_supports_attention_backend": True,
                "config": config,
                "named_modules": lambda self: iter((("layer.attn", attention),)),
            },
        )()

        self.assertEqual(
            _validate_resolved_attention(model, "flash_attention_2"),
            "flash_attention_2",
        )

    def test_loaded_attention_fallback_is_rejected(self):
        config = type("Config", (), {"_attn_implementation": "sdpa"})()
        model = type(
            "Model",
            (),
            {
                "_supports_attention_backend": True,
                "config": config,
            },
        )()

        with self.assertRaisesRegex(RuntimeError, "refusing fallback"):
            _validate_resolved_attention(model, "flash_attention_2")


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

        model = _load_model_exact(
            Loader,
            "example/model",
            composite,
            {
                "dtype": "auto",
                "attn_implementation": "flash_attention_2",
            },
        )

        self.assertEqual(model, "model")
        self.assertIs(seen["config"], composite)
        self.assertEqual(seen["attn_implementation"], "flash_attention_2")
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

    def test_parameter_footprint_reports_weight_size_per_actual_device(self):
        class Parameter:
            def __init__(self, device, count, element_bytes):
                self.device = device
                self.count = count
                self.element_bytes = element_bytes

            def numel(self):
                return self.count

            def element_size(self):
                return self.element_bytes

        model = type(
            "Model",
            (),
            {
                "parameters": lambda self: iter(
                    (
                        Parameter("cuda:0", 10, 2),
                        Parameter("cuda:0", 5, 4),
                        Parameter("cpu", 3, 4),
                    )
                )
            },
        )()

        self.assertEqual(
            _parameter_footprint_by_device(model),
            {
                "cpu": {
                    "tensors": 1,
                    "parameters": 3,
                    "logical_bytes": 12,
                    "resident_bytes": 12,
                },
                "cuda:0": {
                    "tensors": 2,
                    "parameters": 15,
                    "logical_bytes": 40,
                    "resident_bytes": 40,
                },
            },
        )

    def test_meta_parameter_footprint_is_not_reported_as_resident(self):
        class Parameter:
            device = "meta"

            @staticmethod
            def numel():
                return 10

            @staticmethod
            def element_size():
                return 2

        model = type(
            "Model",
            (),
            {"parameters": lambda self: iter((Parameter(),))},
        )()

        self.assertEqual(
            _parameter_footprint_by_device(model)["meta"],
            {
                "tensors": 1,
                "parameters": 10,
                "logical_bytes": 20,
                "resident_bytes": 0,
            },
        )

    def test_placement_summary_prioritizes_actual_mixed_placement(self):
        summary = _placement_summary(
            {"cuda:0": {}, "cpu": {}},
            {"layer.0": "cuda:0", "layer.1": "cpu"},
        )

        self.assertEqual(summary, "GPU (CUDA) + CPU")

    def test_numeric_accelerate_device_map_is_reported_as_cuda(self):
        self.assertEqual(_placement_summary({}, {"": 0}), "GPU (CUDA)")

    def test_progress_reports_true_remaining_work(self):
        message = _sampling_progress_message(
            context_position=1,
            context_count=5,
            context_id="ctx-2",
            context_completed=3,
            samples_per_context=10,
            total_completed=13,
            total_samples=50,
            unfinished_contexts=4,
            sampled_this_invocation=3,
        )

        self.assertIn("13/50 completed overall (26.0%)", message)
        self.assertIn("37 rollouts still require model sampling", message)
        self.assertIn("4 unfinished contexts/prompts", message)
        self.assertIn("3 sampled this invocation", message)

    def test_final_progress_has_zero_remaining(self):
        message = _sampling_progress_message(
            context_position=4,
            context_count=5,
            context_id="ctx-5",
            context_completed=10,
            samples_per_context=10,
            total_completed=50,
            total_samples=50,
            unfinished_contexts=0,
            sampled_this_invocation=7,
        )

        self.assertIn("50/50 completed overall (100.0%)", message)
        self.assertIn("0 rollouts still require model sampling", message)
        self.assertIn("0 unfinished contexts/prompts", message)

    def test_checkpoint_scan_counts_only_valid_completed_batches(self):
        first_batch = (
            SampledSequence("one", 0, (1,), "max_length"),
            SampledSequence("one", 1, (2,), "max_length"),
        )
        last_batch = (SampledSequence("two", 4, (3,), "max_length"),)

        class Store:
            completed = {(0, 0): first_batch, (1, 4): last_batch}

            def load_batch(
                self,
                context_position,
                context_id,
                start,
                expected_count,
            ):
                batch = self.completed.get((context_position, start))
                if batch is not None:
                    self.test_case.assertEqual(len(batch), expected_count)
                    self.test_case.assertTrue(
                        all(sample.context_id == context_id for sample in batch)
                    )
                return batch

        store = Store()
        store.test_case = self
        updates = []
        batches, counts = _scan_checkpointed_batches(
            store,
            (PromptContext("one", "a"), PromptContext("two", "b")),
            samples_per_context=5,
            batch_size=2,
            progress=lambda scanned, total, completed: updates.append(
                (scanned, total, completed)
            ),
        )

        self.assertEqual(set(batches), {(0, 0), (1, 4)})
        self.assertEqual(counts, (2, 1))
        self.assertEqual(updates[-1], (6, 6, 3))

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
            attention_implementation="sdpa",
            device_map="auto",
        )

        with self.assertRaises(ValueError):
            generate_samples(settings, (PromptContext("unconditional", None),))


if __name__ == "__main__":
    unittest.main()
