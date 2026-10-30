# Shipped datasets: sources and licences

Built by `scripts/demo_datasets.py`; each file keeps its source's licence.

## `retail_synthetic.csv` — Роздрібні продажі (синтетичні)

- Source: SIG, синтетичні дані з відомою істиною
- Licence: згенеровано (insight_graph_service/core/demo.py)
- Rows: 5,000

## `california_housing.csv` — Житло Каліфорнії (мапа)

- Source: Pace R. K., Barry R. Sparse spatial autoregressions, 1997 (StatLib)
- Licence: StatLib (Pace & Barry, 1997), вільне використання з посиланням
- Rows: 20,640

## `sf_fire_calibration.csv` — Пожежі Сан-Франциско: 2017-01 … 2019-06

- Source: DataSF «Fire Incidents» (wr8u-xric) через Kaggle san-francisco/sf-fire-data-incidents-violations-and-more
- Licence: Open Data Commons Public Domain Dedication and License (PDDL) 1.0
- Rows: 80,567; 2017-01-01 … 2019-06-30
- Prepared: scripts/sf_fire_benchmark.py: дублікати прибрано, 2017-01 … 2020-08, точки в межах міста, тривалості з часових міток, коди NFIRS згруповано

## `sf_fire_holdout_a.csv` — Пожежі Сан-Франциско: 2019-07 … 2020-02

- Source: DataSF «Fire Incidents» (wr8u-xric) через Kaggle san-francisco/sf-fire-data-incidents-violations-and-more
- Licence: Open Data Commons Public Domain Dedication and License (PDDL) 1.0
- Rows: 22,118; 2019-07-01 … 2020-02-29
- Prepared: scripts/sf_fire_benchmark.py: дублікати прибрано, 2017-01 … 2020-08, точки в межах міста, тривалості з часових міток, коди NFIRS згруповано

## `sf_fire_holdout_b.csv` — Пожежі Сан-Франциско: 2020-03 … 2020-08

- Source: DataSF «Fire Incidents» (wr8u-xric) через Kaggle san-francisco/sf-fire-data-incidents-violations-and-more
- Licence: Open Data Commons Public Domain Dedication and License (PDDL) 1.0
- Rows: 13,178; 2020-03-01 … 2020-08-31
- Prepared: scripts/sf_fire_benchmark.py: дублікати прибрано, 2017-01 … 2020-08, точки в межах міста, тривалості з часових міток, коди NFIRS згруповано

## `sf_fire_calls.csv` — Виклики пожежної служби Сан-Франциско, 2016–2018

- Source: DataSF, Fire Department and EMS Dispatched Calls for Service (nuek-vuh3), через Kaggle jacopoferretti/san-francisco-fire-department-calls-dataset
- Licence: Open Data Commons PDDL 1.0 (DataSF); копія Kaggle — CC0
- Rows: 48,856; 2016-11-01 … 2018-10-31
- Attribution: San Francisco Fire Department / DataSF, Fire Department and EMS Dispatched Calls for Service (PDDL); Kaggle upload by jacopoferretti.
- Prepared: 2016-11 … 2018-10; шоста частина викликів за хешем номера; один рядок на виклик — перший направлений підрозділ, кількість підрозділів і затримки виїзду по всіх; рідкісні рівні — Other

## `chicago_crashes.csv` — ДТП у Чикаго, 2023–2024

- Source: City of Chicago Data Portal, Traffic Crashes (85ca-t3if, u6pd-qa9d), через Kaggle hientrandai/chicago-traffic-crashes-dataset
- Licence: City of Chicago Data Portal Terms of Use; копія Kaggle позначена CC0
- Rows: 37,027; 2023-10-01 … 2024-09-30
- Attribution: City of Chicago / Chicago Police Department (E-Crash), data.cityofchicago.org; Kaggle copy by hientrandai.
- Prepared: 2023-10 … 2024-09; третина аварій за хешем ідентифікатора; учасники й вік водіїв з таблиці людей; рідкісні рівні — Other

## `chicago_schools.csv` — Школи Чикаго, 2023–24 (контекстний шар)

- Source: City of Chicago Data Portal, Chicago Public Schools - School Locations SY2324 (mv87-m4mi)
- Licence: City of Chicago Data Portal Terms of Use
- Rows: 649
- Attribution: City of Chicago / Chicago Public Schools, data.cityofchicago.org.
- Prepared: усі 649 шкіл; точка з Lat/Long; тип — ES/HS

## `nyc_collisions.csv` — ДТП у Нью-Йорку, 2023–2024

- Source: NYC Open Data (NYPD), Motor Vehicle Collisions - Crashes (h9gi-nx95), через Kaggle sahirmaharajj/motor-vehicle-collisions-crashes
- Licence: NYC Open Data Terms of Use; копія Kaggle позначена Apache 2.0
- Rows: 44,696; 2023-03-01 … 2024-02-29
- Attribution: NYPD Motor Vehicle Collisions - Crashes, NYC Open Data (h9gi-nx95); Kaggle mirror by sahirmaharajj.
- Prepared: 2023-03 … 2024-02; аварії з точкою в межах міста; половина за хешем COLLISION_ID; тип транспорту — 8 класів; тип місця з полів вулиць (самі вулиці не включено)

## `us_accidents.csv` — ДТП у США: округ Даллас, 2021–2022

- Source: Moosavi et al., US Accidents (March 2023 release), через Kaggle sobhanmoosavi/us-accidents
- Licence: CC BY-NC-SA 4.0 (некомерційне використання, на тих самих умовах)
- Rows: 37,580; 2021-01-01 … 2022-12-31
- Attribution: Moosavi, Samavatian, Parthasarathy, Ramnath, "A Countrywide Traffic Accident Dataset", 2019 (arXiv:1906.05409); Moosavi, Samavatian, Parthasarathy, Teodorescu, Ramnath, "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights", ACM SIGSPATIAL 2019 (arXiv:1909.09638). This sample is redistributed under CC BY-NC-SA 4.0.
- Prepared: округ Даллас, джерело Source1, 2021–2022; повторні трансляції одного інциденту злито (той самий початок і вулиця: найпізніше завершення); неправдоподібні значення — порожні

## `uk_road_safety.csv` — ДТП у Лондоні, 2017

- Source: Department for Transport, Road Safety Data (STATS19), через Kaggle tsiaras/uk-road-safety-accidents-and-vehicles
- Licence: Open Government Licence v3.0 (DfT STATS19); копія Kaggle — ODbL 1.0 / DbCL 1.0
- Rows: 26,746; 2017-01-01 … 2017-12-31
- Attribution: Contains public sector information licensed under the Open Government Licence v3.0 (Department for Transport, STATS19).
- Prepared: Metropolitan Police, 2017; точки в межах Великого Лондона; один рядок на пригоду; рідкісні рівні — Other

## `us_wildfires.csv` — Лісові пожежі Каліфорнії, 2016–2018

- Source: USDA Forest Service, FPA FOD 6th edition (RDS-2013-0009.6), через Kaggle behroozsohrabi/us-wildfire-records-6th-edition
- Licence: CC0 (Kaggle); USDA Forest Service Research Data Archive, суспільне надбання з посиланням на DOI
- Rows: 26,884; 2016-01-01 … 2018-12-31
- Attribution: Short, K.C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 [FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6
- Prepared: Каліфорнія, 2016–2018; повтори однієї пожежі злито до найменшого FOD_ID; клас розміру не включено (він ділить площу на смуги, тож лише повторював би її); рідкісні рівні — other
