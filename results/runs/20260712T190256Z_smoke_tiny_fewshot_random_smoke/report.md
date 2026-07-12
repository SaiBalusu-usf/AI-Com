# Run report: smoke_tiny_fewshot_random_smoke

## Run info

| field | value |
| --- | --- |
| name | smoke_tiny_fewshot_random_smoke |
| system | fewshot |
| config_path | configs/smoke/tiny_fewshot.yaml |
| overrides | [] |
| data | data/fixtures/mini.jsonl |
| split | all |
| n_rows | 200 |
| seed | 13 |
| timestamp | 20260712T190256Z |
| git_commit | 20947eb |

## faithfulness

| metric | value |
| --- | --- |
| n | 200 |
| hallucination_rate | 0.2200 |
| omission_rate | 0.8987 |
| faithfulness_score | 0.7800 |
| per_slot.attribution.claims | 1 |
| per_slot.attribution.false_claims | 1 |
| per_slot.attribution.precision | 0.0000 |
| per_slot.boundary.claims | 25 |
| per_slot.boundary.false_claims | 22 |
| per_slot.boundary.precision | 0.1200 |
| per_slot.runs.claims | 6 |
| per_slot.runs.false_claims | 6 |
| per_slot.runs.precision | 0.0000 |
| per_slot.wicket.claims | 21 |
| per_slot.wicket.false_claims | 20 |
| per_slot.wicket.precision | 0.0476 |
| violations_by_type.boundary | 22 |
| violations_by_type.wicket | 18 |
| violations_by_type.runs | 6 |
| violations_by_type.dismissal_type | 2 |
| violations_by_type.attribution | 1 |
| recall.wicket_mentioned | 0.1154 |
| recall.boundary_mentioned | 0.0943 |
| ner | unavailable (spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.) |
| examples.violations | [{'text': 'hand GH ake i 99 added 42 tid Ambrose 87 Ver as 44 ( Nesbit GONE U six 72 X Holloway ed taken iew', 'violations': [{'slot': 'boundary', 'claim_slot': 'boundary', 'claimed': 6, 'actual': 0, 'evidence': 'six/maximum'}]}, {'text': '15 33 ckwo sive gate GLE lbw Ver fi 57 Ock the 18 g the Barretto Gaikwad 61 es sive get Saffron pla Ockwell', 'violations': [{'slot': 'wicket', 'claim_slot': 'wicket', 'claimed': 'lbw', 'actual': 'no wicket', 'evidence': 'lbw'}]}, {'text': 'ra Lakhani ays 08 Ver shat beautif Ob Ock Nesbitt nificent Into sol 44 ande Qui lbw 150 170 Zutshi ster 128 Whit cely', 'violations': [{'slot': 'wicket', 'claim_slot': 'wicket', 'claimed': 'lbw', 'actual': 'no wicket', 'evidence': 'lbw'}]}, {'text': '15 BYES2 keeps aw remo bowler Ver se 57 104 Vipers three uri ne CAU Sar overst Four Greaves ake 05 60 Far TWO', 'violations': [{'slot': 'boundary', 'claim_slot': 'boundary', 'claimed': 4, 'actual': 0, 'evidence': 'four'}]}, {'text': 'hand 37 Fal Everton Ob K ani 169 mo Quick 82 erplay dot board 194 1 si 104 two boundary ter 26 177 ugh', 'violations': [{'slot': 'boundary', 'claim_slot': 'boundary', 'claimed': 'any', 'actual': 0, 'evidence': 'boundary'}]}, {'text': 'squ S that FOU bowler caught ha 87 65 nificent Lakhani 150 or crea dekar 52 trapped get 158 phase cross Naidu CAU', 'violations': [{'slot': 'wicket', 'claim_slot': 'wicket', 'claimed': 'caught', 'actual': 'no wicket', 'evidence': 'caught'}]}, {'text': 'hand keep has Ver as Vipers overst Four erb off Ha more Nomads 03 cross ni BY ly z Qures GLE Ob id 112', 'violations': [{'slot': 'boundary', 'claim_slot': 'boundary', 'claimed': 4, 'actual': 2, 'evidence': 'four'}]}, {'text': 'hand ce Qui I ter 21 od ath bat Kulkarni THREE Hazarika i CAU boundary 107 Uthapp 170 powerplay 79 ton | Kings CAUGH', 'violations': [{'slot': 'boundary', 'claim_slot': 'boundary', 'claimed': 'any', 'actual': 2, 'evidence': 'boundary'}]}, {'text': 'Advani WICKET S climbs Ver crowd bri Vi 82 142 93 67 by _in_hand fle Ever rose Uthapp pressu nudges ds RRR 135 am', 'violations': [{'slot': 'wicket', 'claim_slot': 'wicket', 'claimed': 'unspecified', 'actual': 'no wicket', 'evidence': 'WICKET'}]}, {'text': "Harborview croix ( Delacroix mid ' dow sol they NE 76 97 ing 29 109 196 Ni caught cle fielder nudges c Cummin Ver", 'violations': [{'slot': 'wicket', 'claim_slot': 'wicket', 'claimed': 'caught', 'actual': 'no wicket', 'evidence': 'caught'}]}] |
| examples.omissions | [{'text': 'zro S 158 Whitfield Fitzro Deshpande mp 89 Hazarika pressure remo mb vani Basalt ry Wide lander be 6 bowler rope 48 view ly', 'omissions': ['four unmentioned']}, {'text': "Zutshi 166 overst sive 181 aw ks . wate Barret _hand OU 08 overst aw ' 132 ha 132 Zut Nomads 122 phase No", 'omissions': ['wicket unmentioned']}, {'text': 'lander Nor ed taken e BYES Whit ugh get 02 25 20 08 58 wide squeezing co 95 9 That el 104 201 ik', 'omissions': ['four unmentioned']}, {'text': 'fin 97 mp end crowd Uthapp Cra ter str 130 65 ET 201 sol Whit wkts_in_hand ad lets 186 BYES Kulkar 102 148 aw', 'omissions': ['wicket unmentioned']}, {'text': 'defend Verlander 18 84 69 29 ter BOWLE No 65 wani Superb 14 scampers NE 87 65 nificent ay 40 Qu Lakha ke 157', 'omissions': ['wicket unmentioned']}, {'text': 'am Vaswani de 60 Tre o powerplay lo LE res rea 91 Ver bri bu el tra ore pp dekar magnificent 128 EE LE', 'omissions': ['four unmentioned']}, {'text': 'get Nesbit 128 clears tar N_OUT 17 nder cro come Lakha they 161 ile Two 196 hi 60 scampers rose 59 ika /', 'omissions': ['six unmentioned']}, {'text': 'fin Nor THR lapp Quill Dela Zutshi kar 104 mag ver Thampi ings Ad climb mid bri 54 utely beautif F tlo dekar get', 'omissions': ['six unmentioned']}, {'text': 'lapp Dela Uthapp 13 Gaikwad 83 37 Zutshi 44 117 target end powerplay Ock Des chance ou c 188 phase phase oriou BAL kts', 'omissions': ['four unmentioned']}, {'text': 'Int phase Sar mag ter ni ugh cro 66 60 gather ri sing Dela out mome W nu phase NGLE Lakha Smart ee ep', 'omissions': ['four unmentioned']}] |
| reference_baseline.hallucination_rate | 0.0000 |
| reference_baseline.omission_rate | 0.2025 |

## excitement

| metric | value |
| --- | --- |
| reference_validation.spearman_rho | 0.8064 |
| reference_validation.n | 200 |
| reference_validation.mode | heuristic |
| reference_validation.fallback_reason | could not load excitement model 'j-hartmann/emotion-english-distilroberta-base': We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'. |
| reference_validation.per_phase.powerplay | 0.6733 |
| reference_validation.per_phase.middle | 0.8035 |
| reference_validation.per_phase.death | 0.8561 |
| reference_validation.passed | True |
| calibration.spearman_rho | 0.0803 |
| calibration.n | 200 |
| calibration.mode | heuristic |
| calibration.fallback_reason | could not load excitement model 'j-hartmann/emotion-english-distilroberta-base': We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'. |
| calibration.per_phase.powerplay | -0.0030 |
| calibration.per_phase.middle | 0.1062 |
| calibration.per_phase.death | 0.0623 |
| calibration.passed | True |

## surface

| metric | value |
| --- | --- |
| bleu | 0.0364 |
| rouge_l | 0.0196 |
| bert_score | unavailable (bert-score could not run (roberta-large weights or torch backend missing): We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.) |

## diversity

| metric | value |
| --- | --- |
| distinct_1 | 0.7098 |
| distinct_2 | 0.9911 |
| self_bleu | 0.0118 |
| repetition_4gram_rate | 0.0000 |
| n_generations | 200 |

## latency

| metric | value |
| --- | --- |
| n_timed | 200 |
| p50_ms | 61.0220 |
| p95_ms | 73.6802 |
| mean_ms | 61.3923 |
| tokens_per_sec | 373.9880 |
| hardware.platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| hardware.cpu_count | 4 |
| hardware.torch_device | cpu |

## Metrics not computed

- `faithfulness.ner`: spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
- `surface.bert_score`: bert-score could not run (roberta-large weights or torch backend missing): We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
