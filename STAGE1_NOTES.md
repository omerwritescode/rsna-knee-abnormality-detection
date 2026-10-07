# RSNA Knee — Stage 1 labeler notes (7 Oct 2026)

Addendum to RSNA_Knee_Progress.md; overrides it for Stage 1.

## Split (dev 20 / held-out 38)
- Seeded search (seed 2026, 200k draws) for the most balanced split, >=2 positives per label per side.
- Dev positives: ACL 8, MCL 3, MM 9, LM 7, MOA 5, LOA 4, PFOA 8, Eff 12, Syn 10, Baker's 4, Cont 7, Fx 6.
- Made without study_meta. If the laptop run prints "0 with >1 study", the split is identical to data/labels/split.csv here.
- Only the dev 20 reports have been read. Held-out reports are untouched.

## Key finding: the 58 labels were probably read from the images
8 of 20 dev studies have at least one label that contradicts the report
("no fracture or bone contusion" -> Contusion=1; normal report -> Baker's=1;
"moderate joint effusion" -> Effusion=0; "medial meniscus tear" -> MM=0).
- Held-out score has a ceiling. Use it to catch systematic definition errors, not to chase every miss.
- Synovitis is positive in 10/20 dev studies but named in ~4 reports; it tracks substantial effusion.
  Prompt v1 marks moderate/large effusion or hemarthrosis without synovial comment as uncertain Synovitis.

## Definitions in prompt v1
- ACL: present = complete/subtotal/high-grade partial; uncertain = low-grade/interstitial, sprain; absent = mucoid degeneration.
- MCL: present = grade II-III / partial or complete tear; uncertain = low-grade partial; absent = grade I, periligamentous edema.
- Meniscus: present = any tear incl. root, ramp; absent = grade I-II degeneration, meniscocapsular signal, extrusion alone.
- OA: present = named OA, osteophytes, narrowing, grade 3-4/full-thickness, "incipient", tricompartmental; uncertain = grade II; absent = superficial/grade I.
  PF OA also: chondromalacia patellae III-IV, full-thickness focal patellar/trochlear defects.
- Effusion: small and above, hemarthrosis; absent = minimal/trace/physiological.
- Baker's: any size; trace = uncertain.
- Contusion: bruise, kissing contusion, impaction injury; uncertain = unexplained post-trauma edema; absent = edema next to fracture/OA/insufficiency.
- Fracture: any incl. insufficiency/subchondral/avulsion; absent = osteochondral defect, loose body, bruise only.
