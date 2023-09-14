# Benchmarking LLM

### Setup model/server environment

Follow step 1 from the [Triton + vLLM tutorial](https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/vLLM/README.md).

### Prefill

In this benchmarking scenario, we want to measure the effect of input prompt size on first-token latency. We issue single request to the server of fixed input sizes and request the model to compute at most one new token. This essentially means one pass through the model.

#### 1. Run these commands

```bash
PATH_TO_MODEL_PY="model_repository/vllm/1/model.py"
MAX_TOKENS=1
sed -i "128s/.*/\ \ \ \ \ \ \ \ params_dict[\"max_tokens\"] = ${MAX_TOKENS}/" ${PATH_TO_MODEL_PY}
```

#### 2. Follow step 2 from the [Triton + vLLM tutorial](https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/vLLM/README.md).

#### 3. Generate prompts input data JSON

```bash
echo '
{
    "data": [
        {
            "PROMPT": [
                "resink transversomedial pharyngopathy postmineral myelosyphilis silverer evincement phrygium punnigram imminution environmental sleepify nope wauken indignance knotwort apocodeine escortee dogwatch eaglewood unbrotherliness mulse dermobranchiata typhic poststertorous indevout anatomicopathologic unimpenetrable hoggy urrhodin Dioecia unchapter nonumbilicate zwitterionic apportionable ferulic statefulness pharyngotonsillitis Mimulus recce mutinously reboant marshwort lupoid chromatophilic lauder nirles esthesiometer semisocial unbeing kangaroo takosis inconvertibility anesthetist rumorproof thoracoscopy euphorbium bizet song dolichocephali platemaker vesicupapular electroforming dilatingly meethelp loincloth avowably counterindicate treacliness Epigonus airmark polarography precomposition lemography Apinage Taal logology probeer randomization poditic individualize castigate Biloculina overscrub koolah weetless erased layery discontinuee anaphylatoxin unwounded personalism howitzer hexahydroxy koku reamer tonguiness microgametocyte baba ludefisk novelwright swinehull Odonata indefinable faineance nidologist supracargo beriberic betso archheart snary Viminal Pygopodidae acetylenediurein asphalt preimpress fountainlet bejel unpictorially heliophyte chimopeelagic warison antivaccinist overtwine preremove nerval bufonite eradicator turtling winrace psychographic impalpably amygdalase Octogynia brimming grist casave brazilein afluking meliceris portative unsteck Madelon barramunda optotechnics metapterygium unromanticalness Jacobinism pricklingly blameless elderhood committeewoman comicocynical aggrate stentoronic flatwise bipyridyl untastable aegerian unmistrusted quadrigamist Meleagrinae helvite neuralist Swietenia unpleadable colorably mogilalism consequently atamasco inhospitality noncarnivorous counterruin gryposis ringe displeasedly incenter gallycrow whincow repudiationist unagile chaplain bekerchief subproduct pointingly Physonectae bumpingly hateful endogenous facticide velours carmoisin reaccomplish protistic recuperance tech withywind Galen Slavistic escropulo deglutination hydramnios Amphion beguilement glottiscope propagation entrancement disbelief goatlike Tanyoan thecium deforciant coachwhip enviableness duroquinone smirchy whisky forcing homosystemic underact autosyndesis sybaritism scorching testiere nonporphyritic cephalhematoma oxyquinoline azo scrimshorn unreeling burnt kilocycle lactenin Decimus patter jetbead Pygidium bitterroot thoke septated trinodal qualitied gotten unclassable Akhissar wholewise curse organophyly teleseme heptitol whitehass asclepiadeous labionasal presumptuous triketo thrombolymphangitis spokeswomanship unprejudicial ungoverned nonrectangular pleocrystalline slurbow unhandsome scoliotic phreatophyte criminalistics imitability zygozoospore nonliability disafforestation epigenetic Aves schistaceous verbomania epitenon proscriber histology propulsion elaidinic driftless upcover duteous distasteful dermatoxerasia Kaibartha spydom tonsor paedogenesis anticipatory unastonished Blackbeard gradient metachemistry caravan circumnutate infract adenia louse koel tributyrin lifey Desmidiaceae vinelet ingrowth uniovulate toying nonretentive conjunct pinnaglobin vastity appendicle redecline trekker hereby dicatalexis slackerism inaxon tribase cryptostome nonpresbyter breezily opusculum methought yarl fringent forsooth plicater balneotherapeutics uredosporic perceptive embouchure heterolysis imperence perfervidness pobs thorax perikronion charlatanic Bonapartism whilom outferret kelty macrospore predisposedly squawk extrafoliaceous inveteracy obtect decoyer Pelecaniformes dodecane gambet electrodynamism henter sunless burroweed busket trepidatory fermerer prewound thrifty twibil amateur myasthenia goave toolmark ornithon geminately unrhymed Serridentines presbyopia unoperably preventer salten grimily conduplicated theomania gyromagnetic antimycotic malacanthid sensationistic vibraculoid eater zig bordello hounding outweary hoyle nonrendition potlike surflike rubification dulcifluous Saturnalian unconfidence Apneumona hedgy subharmonic undisputed monotypic pontifex Phalarism precursive uncock echinoderm antu rollick natricine presuperintendence pinnaclet precondemnation Atheriogaea volumescope Austrophilism stinking wildness noncoloring spaying somniloquy xi hierogrammatical winer ironback tarnside lampers handcraft glossophagine philophilosophos nonconcludent overaccumulate disbutton kinetomer thermostimulation stenogastric ovoviviparously recept firetop roughroot Muncerian prefiction Ovinae reactivity oncin pointer absolve unaccommodatingly telson ayelp rebegin unhomely Octavian scope Pentelic revocability juvenal spinobulbar erinaceous hield anaglyph strongylid strangling kala fibroplastic adactyl Pauline undispellable Frederick amylopsin informative Sisseton roominess unsurpassableness painstaker saturator laryngoscopical stereophotographic washbasin functionarism absorbability enscroll scunner masting lionet unbumptious stockishness prechemical nonmythical apache force isomastigate orthophosphate Palaeomastodon brachypyramid abscession acquisitum reputationless praisefully grama trapped Somal disturn menorrhagia faltering Yquem invective Hafgan isobarbaloin phototrichromatic ectomorphy rollickingness preponderance nonprobable counterreckoning unenforcedly Saratogan minienize limby lovelorn presartorial Chaetophoraceae intenable satisfying columnization brotulid interenjoy subtriplicated Vaudois Dioscorea Brachiata unpunishable Latvian siccity blossomtime Castalia dephlegmator apagogically Aglypha withdrawment hemoproctia unfailableness inventorial Pyrrhic barbiturate undetermination osteoscope lutulence Rajah tortricoid subglacially porwigle complacent archagitator sterigmatic hydrocycle misliken powerhouse manumission Dardistan oosporic vestal baller heterochronic flyless cuboidal adenology miscellaneous imperceptibly decohesion Babel thaliacean underivable misexample hypersophisticated Cucurbitaceae cherubic esophagotomy Suomic staghorn hysteric quadrumane keratocentesis middlings alley Delicious chymotrypsin cancroid aweary capersome Ashkenazim ventripotential Chlamydoselachus dithioic weeze ruck overhover stabulate littleneck duplone bulkhead niellated bellite samsara diligentness scritch amuck studbook guijo certifiableness tormentful milliare repromulgate synesthesia whitecoat osmometric periductal psorospermosis purificator untrochaic Jeremian copulate ratable dislodgement proferment evangelary overdevotedly lickspittling atrocity supracaudal uncompassionate nonsparking shaftfoot attemperation unentrance mispossessed dumpy strangership hygrodeik foundery scenic purchase scorch preaffiliation Cossaean tungstate hecte ureometer syllabatim wireless Zapoteco notarikon acroasphyxia endosalpingitis humpback unwist pedigerous peacemaking foremasthand annodated multicarinated Elaps seedbox loaferish proprietage Eumenes monochlor decarhinus ambry"
            ],
            "STREAM": [
                true
            ]
        }
    ]
}
' > prompts.json
```

#### 3. Run PA

```bash
perf_analyzer \
    -m vllm \
    -i grpc \
    --async \
    --streaming \
    --input-data=prompts.json \
    --profile-export-file=profile_export.json \
    --measurement-mode=count_windows \
    --measurement-request-count=10 \
    --stability-percentage=999
```

#### 4. Calculate average time to first token

```bash
python3 -c 'import json
f = open("profile_export.json")
requests = json.load(f)["experiments"][0]["requests"]
latencies = [r["response_timestamps"][0] - r["timestamp"] for r in requests]
avg_latency_s = sum(latencies) / len(latencies) / 1000000000
print("Average time to first token: " + str(avg_latency_s) + " s")'

# Average time to first token: 0.672695455483871 s
```

#### 5. Repeat steps 3-4 with different prompt lengths to measure effects of initial prompt size (prefill) on first-token latency.

### Generation

In this benchmarking scenario, we want to measure the effect of input prompt size on token-to-token latency. We issue single request to the server of fixed input sizes and request the model to compute a fixed amount of tokens.

#### 1. Run these commands

```bash
PATH_TO_MODEL_PY="model_repository/vllm/1/model.py"
MAX_TOKENS=256
sed -i "128s/.*/\ \ \ \ \ \ \ \ params_dict[\"max_tokens\"] = ${MAX_TOKENS}/" ${PATH_TO_MODEL_PY}
```

#### 2. Follow step 2 from the [Triton + vLLM tutorial](https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/vLLM/README.md).

#### 3. Generate prompts input data JSON

```bash
echo '
{
    "data": [
        {
            "PROMPT": [
                "Hello, my name is"
            ],
            "STREAM": [
                true
            ]
        }
    ]
}
' > prompts.json
```

#### 3. Run PA

```bash
perf_analyzer \
    -m vllm \
    -i grpc \
    --async \
    --streaming \
    --input-data=prompts.json \
    --profile-export-file=profile_export.json \
    --measurement-mode=count_windows \
    --measurement-request-count=10 \
    --stability-percentage=999
```

#### 4. Calculate average time to first token

```bash
python3 -c 'import json
f = open("/workspace/tmp/profile_export.json")
requests = json.load(f)["experiments"][0]["requests"]
latencies = []
for request in requests:
    prev_response = request["response_timestamps"][0]
    for response in request["response_timestamps"][1:]:
        latencies.append(response - prev_response)
        prev_response = response
avg_latency_s = sum(latencies) / len(latencies) / 1000000000
print("Average token-to-token latency: " + str(avg_latency_s) + " s")'

# Average token-to-token latency: 0.003090155677419355 s
```

#### 5. Repeat steps 3-4 with different prompt lengths to measure effects of initial prompt size (prefill) on token-to-token latency (generation).
