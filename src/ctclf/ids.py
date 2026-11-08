import re


def patient_id(filename):
    """Work out the patient ID from a COVIDx CT slice filename.

    Each source hospital names its files differently, so each needs its own rule.
    The IDs match the 'patient id' column in metadata.csv, except for the
    underscore-style radiopaedia files, which aren't in the metadata at all.
    """
    stem = filename.replace(".png", "")
    if stem.startswith(("NCP_", "CP_", "Normal_")):      # CNCB
        return "_".join(stem.split("_")[:2])
    if "_patient" in stem:                               # COVID-CTset
        return re.search(r"patient\d+", stem).group()
    if stem.startswith("volume-"):                       # TCIA
        return re.match(r"volume-covid19-A-\d+(_\d)?", stem).group()
    if stem.startswith("LIDC-IDRI-"):                    # LIDC-IDRI
        return "-".join(stem.split("-")[:3])
    if stem.startswith("radiopaedia-"):                  # radiopaedia, dash style
        return "-".join(stem.split("-")[:2])
    if stem.startswith("radiopaedia_"):                  # radiopaedia, underscore style
        return "radiopaedia_" + stem.split("_")[1]
    if re.match(r"A\d{6}-", stem):                       # Stony Brook
        return stem.split("-")[0]
    return stem.rsplit("-", 1)[0]   # STOIC, iCTCF, COVID-CT-MD, COVID-19-CT-Seg


def person_id(pid):
    """Group TCIA scans of the same person.

    TCIA lists a person's separate scans as separate patients
    (volume-covid19-A-0047_0 and _1). For splitting we treat them as one person,
    otherwise the same lungs can end up in both train and test.
    """
    if pid.startswith("volume-"):
        return re.sub(r"_\d$", "", pid)
    return pid