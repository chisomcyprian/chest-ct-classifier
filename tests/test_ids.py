import pytest

from ctclf.ids import patient_id, person_id


@pytest.mark.parametrize("filename, expected", [
    ("NCP_96_1328_0032.png", "NCP_96"),
    ("STOIC-10011-0103.png", "STOIC-10011"),
    ("HUST-Patient1-0060.png", "HUST-Patient1"),
    ("COVIDCTMD-P055-IM0024.png", "COVIDCTMD-P055"),
    ("coronacases_001-0085.png", "coronacases_001"),
    ("normal1_patient167_SR_2_IM00006.png", "patient167"),
    ("volume-covid19-A-0003_ct-0065.png", "volume-covid19-A-0003"),
    ("volume-covid19-A-0001-0177.png", "volume-covid19-A-0001"),
    ("volume-covid19-A-0047_1_ct-0011.png", "volume-covid19-A-0047_1"),
    ("LIDC-IDRI-0068-1.3.6.1.4.1.14519.5.2.1.6279.6001.187108608022306504546286626125-0048.png",
     "LIDC-IDRI-0068"),
    ("radiopaedia-11516-49631-0-0000.png", "radiopaedia-11516"),
    ("radiopaedia_10_85902_3-0054.png", "radiopaedia_10"),
    ("A002304-01-10-1901-NA-CT-ABD-AND-PELVIS-WITH-IV-CONT-76655-2-0001.png", "A002304"),
])
def test_patient_id(filename, expected):
    assert patient_id(filename) == expected


def test_person_id_merges_tcia_scans():
    assert person_id("volume-covid19-A-0047_0") == "volume-covid19-A-0047"
    assert person_id("volume-covid19-A-0047_1") == "volume-covid19-A-0047"
    assert person_id("NCP_96") == "NCP_96"