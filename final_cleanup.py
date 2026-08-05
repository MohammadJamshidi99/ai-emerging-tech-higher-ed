import pandas as pd

IN_FILE = "final_shortlist.csv"
OUT_FILE = "final_shortlist_cleaned.csv"

ENGINEERING_NOISE_TERMS = [
    "flight control", "pid control", "pid tuning", "formation control",
    "sdn", "software defined network", "robot control", "control system",
    "signal processing", "circuit design", "power grid", "wireless network",
    "autonomous vehicle", "drone control", "motor control",
]

# if these are also present, nott flagging
EDUCATION_RESCUE_TERMS = [
    "tutor", "student", "classroom", "curriculum", "lesson", "teaching",
    "course", "learning management", "grading", "quiz", "assignment feedback",
]


def looks_like_engineering_noise(row):
    text = f"{row.get('description', '')} {row.get('readme', '')}".lower()
    if not any(term in text for term in ENGINEERING_NOISE_TERMS):
        return False
    if any(term in text for term in EDUCATION_RESCUE_TERMS):
        return False
    return True


def main():
    df = pd.read_csv(IN_FILE)
    df["likely_false_positive"] = df.apply(looks_like_engineering_noise, axis=1)

    n_flagged = df["likely_false_positive"].sum()
    print(f"{n_flagged}/{len(df)} rows flagged as likely false positives")
    print(df[df["likely_false_positive"]][["full_name", "description"]].head(15).to_string(index=False))

    df.to_csv(OUT_FILE, index=False)
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    main()