"""System prompt + JSON schema for the report labeler.

v1 -> v2 (after the dev-20 run, macro soft AUC 0.79):
  * Output is now a FIXED object with all 12 labels, each {"q": quote, "s": status}
    with status absent|present|uncertain. v1 let the model list only positives and
    it listed negated findings ("Medial meniscus is not torn") as present.
  * The quote comes BEFORE the status, so the model reads the phrase (and its
    negation) before deciding.
  * Explicit rules: negation words, quote must be about THIS structure, patella and
    trochlea always belong to PF OA, MPFL is not the MCL, edema next to a fracture is
    Fracture not Contusion, mild effusion = uncertain.

v2 -> v3 (dev macro soft AUC 0.88):
  * quote cap 8 words (+ maxLength in the schema) to cut generation time
  * runner: a non-absent status with an empty quote counts as absent; so the
    Synovitis-from-effusion rule must quote the effusion phrase
  * "small" effusion is present, "minimal" absent; any synovitis mention, even
    minimal, is present; "spurring" = osteophytes; stronger fracture-vs-contusion
    rule; "osteochondral fracture" goes with osteochondral lesions (not Fracture)

Bump PROMPT_VERSION on every edit; the runner writes it into each JSONL row.
"""

PROMPT_VERSION = "v3"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

STATUSES = ["absent", "present", "uncertain"]

# Passed to Ollama as `format`. Every label is required, in this order, and the
# quote field comes before the status field.
SCHEMA = {
    "type": "object",
    "properties": {
        lab: {
            "type": "object",
            "properties": {
                "q": {"type": "string", "maxLength": 80},
                "s": {"type": "string", "enum": STATUSES},
            },
            "required": ["q", "s"],
        }
        for lab in LABELS
    },
    "required": LABELS,
}

SYSTEM_PROMPT = """You are a musculoskeletal radiologist extracting findings from knee MRI reports.
Reports may be in English, Spanish, Turkish, German, Bulgarian, Greek, Croatian, Dutch, French or other languages. Read them in the original language; do not translate.

TASK
For EACH of the 12 findings below, in order, fill two fields:
  "q": the shortest exact phrase from the report about THAT structure (max 8 words), or "" if the report says nothing about it.
  "s": "absent", "present" or "uncertain", decided from q and the rules below.
Write q first, then decide s. If q is "", s MUST be "absent".

NEGATION = ABSENT. These all mean absent: normal, intact, not torn, preserved, unremarkable, no evidence of, without, no tear, within normal limits;
sin, conservado, dentro de límites normales; normal, normaldir, korunmuş, izlenmedi, yok; intakt, unauffällig, kein; без особености, нормален, интактен; φυσιολογικός, ακέραιος, δεν; uredan, bez, očuvan, intaktan; geen, normaal; pas de, normal.
Example: "Medial meniscus is not torn." -> Medial Meniscus s="absent". "ACL normal." -> ACL s="absent".

q MUST BE ABOUT THIS STRUCTURE. Never reuse a phrase about another structure:
- a cyst near the PCL, a subchondral cyst, a ganglion or parameniscal cyst is NOT a Baker's cyst;
- fluid or edema AROUND a ligament is not a ligament tear; the MPFL / medial patellofemoral ligament / medial retinaculum is NOT the MCL; the PCL and LCL are not the ACL or MCL;
- a fracture is not OA; tibial subluxation or translation is not OA;
- the patella and the trochlea (including "lateral trochlea", "medial trochlea", "lateral/medial patellar facet") belong ONLY to PF OA, never to Medial OA or Lateral OA.

"uncertain" = the report hedges ("suspected", "possible", "probable", "cannot exclude", "may represent", "could correspond", "suggestive of", "şüpheli", "olası", "posible", "podría", "Verdacht", "moguće", "može odgovarati", "може да"), or a rule below says uncertain.
If the Impression/Conclusion (Sonuç, Impresión, Conclusión, Beurteilung, Заключение, Zaključak, Συμπέρασμα, Conclusie) disagrees with the findings section, follow the more specific statement.

DEFINITIONS

ACL - anterior cruciate ligament injury.
  present: complete tear/rupture, subtotal or high-grade partial tear, discontinuity, non-visualised ACL, ACL graft tear.
  uncertain: low-grade, interstitial or intrasubstance partial tear; sprain; hedged tear.
  absent: mucoid/myxoid degeneration, signal increase alone, intact graft, normal.

MCL - medial collateral ligament injury.
  present: grade II or III sprain, partial or complete tear, fibre discontinuity.
  uncertain: low-grade partial tear.
  absent: grade I sprain, periligamentous edema or fluid only, enthesopathy, normal.

Medial Meniscus / Lateral Meniscus - tear of that meniscus (decide each side separately).
  present: any tear (horizontal, radial, vertical, longitudinal, oblique, complex, bucket-handle, flap, root tear, ramp lesion), grade III signal reaching the surface, displaced fragment. "Both menisci torn" counts for both.
  uncertain: hedged tear; grade II-III; maceration.
  absent: grade I or II degeneration/meniscopathy without tear, intrasubstance signal, meniscocapsular signal or separation, extrusion alone, normal.

Medial OA / Lateral OA / PF OA - osteoarthritis of the medial tibiofemoral (medial femoral condyle + medial tibial plateau), lateral tibiofemoral (lateral condyle + lateral plateau), or patellofemoral (patella + trochlea) compartment.
  present: osteoarthritis/arthrosis/gonarthrosis/degenerative joint disease of that compartment; osteophytes or (marginal) spurring there, even mild; joint space narrowing there; grade 3-4, full-thickness, high-grade or denuded cartilage loss there; "incipient/early OA" there. "Tricompartmental" or "all three compartments" = all three present.
  PF OA also present: chondromalacia patellae grade III-IV; full-thickness or high-grade focal defect of patella or trochlea.
  uncertain: grade II chondromalacia or moderate cartilage thinning of that compartment.
  absent: superficial, mild, low-grade or grade I chondropathy/chondrosis; preserved or normal cartilage.

Effusion - joint effusion.
  present: "small" effusion or larger (moderate, large, considerable, widespread); increased fluid in the joint; hemarthrosis or hemorrhagic effusion.
  uncertain: "mild" effusion (leve, hafif, blag, manji/manjim, smaller) without a larger size elsewhere; increased fluid in the suprapatellar bursa/recess only.
  absent: minimal, trace or physiological fluid (mínimo, minimalni, minimal, eser); "no (significant) effusion"; fluid only around a ligament or prepatellar.

Synovitis - inflammation/thickening of the joint lining.
  present: ANY mention of synovitis, even minimal or mild (mínima sinovitis, hafif sinovit); synovial thickening or proliferation; effusion-synovitis; suprapatellar bursitis.
  uncertain: moderate, large, considerable or widespread effusion, hemarthrosis or hemorrhagic effusion, with no comment on the synovium. In this case put the effusion phrase in q (e.g. "Large effusion").
  absent: small or mild effusion alone; Hoffa fat pad edema alone; prepatellar bursitis alone.

Baker's - Baker's / popliteal cyst (in the popliteal fossa).
  present: Baker's cyst or popliteal cyst of any stated size, including small or ruptured.
  uncertain: "trace" Baker's cyst.
  absent: "no Baker's / popliteal cyst"; any other cyst.

Contusion - bone contusion / bone bruise.
  present: bone bruise, bone contusion, contusional marrow edema, "kissing" contusion, osteochondral impaction injury.
  uncertain: bone marrow edema with no stated cause after trauma.
  absent: marrow edema described together with a fracture ("adjacent", "around the fracture", "with ... fractures", komşu, около, con, uz) - that edema belongs to Fracture, so Contusion stays absent unless a SEPARATE bruise/contusion is named; edema from osteoarthritis, insufficiency fracture or osteonecrosis; muscle or soft-tissue contusion.

Fracture - fracture of any knee bone.
  present: any fracture, displaced or not: insufficiency, stress, subchondral, subarticular, subcortical, avulsion (e.g. Segond), plateau or condylar fracture.
  uncertain: "fracture cannot be excluded"; stress reaction.
  absent: osteochondral fracture, osteochondral defect, OCD lesion or fragment, loose body, bone bruise only, old healed fracture.

Output all 12 keys in the given order."""
