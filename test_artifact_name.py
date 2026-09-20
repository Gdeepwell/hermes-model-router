"""A write verb that names an artifact must not contradict a [spark] label."""

import unittest

import model_router as router


GOAL = (
    "[spark] Produce a read-only evidence map for the premium public-booking "
    "initial selection-order feature. Inspect the existing typed storefront "
    "config defaults/normalizers, the three premium editors, admin save/update "
    "API and public DTO/API contracts."
)


class ArtifactNameTests(unittest.TestCase):
    def test_named_endpoint_stays_read_only(self):
        self.assertIs(router._is_spark_read_only_work(GOAL), True)

    def test_slash_cluster_before_artifact_noun_is_a_name(self):
        self.assertIs(router._is_spark_read_only_work(
            "[spark] Map the create/delete endpoints and the DTO shape."
        ), True)

    def test_verb_bound_to_artifact_noun_is_a_name(self):
        self.assertIs(router._is_spark_read_only_work(
            "[spark] Inspect the update handler, the resolver and the normalizers."
        ), True)

    def test_instruction_carrying_the_artifact_noun_is_still_a_write(self):
        self.assertIs(router._is_spark_read_only_work(
            "[spark] Update the payment API to v2 and redeploy."
        ), False)
        self.assertIs(router._is_spark_read_only_work(
            "[spark] Implement the save/update API for the booking form."
        ), False)
        self.assertIs(router._is_spark_read_only_work("[spark] Add a create endpoint."), False)
