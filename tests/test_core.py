"""Tests for Raksha."""
from src.core import Raksha
def test_init(): assert Raksha().get_stats()["ops"] == 0
def test_op(): c = Raksha(); c.detect(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Raksha(); [c.detect() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Raksha(); c.detect(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Raksha(); r = c.detect(); assert r["service"] == "raksha"
