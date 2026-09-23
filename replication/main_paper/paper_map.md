# Manuscript artifact map

Checks concern displayed content and identified saved evidence. They do not certify every original cache or historical execution. Exact delivered notebooks have static validation only.

## tab:splitqc — Final sample and leakage audit. $N$ is the common-complete analysis sample.

Approved groups: Toronto 7,262; facial age 9,658; rice 4,794. Observation and test counts unchanged. Toronto original constructor not claimed.

- Source: appendix/appendix.tex, line 6
- Generator: Inline manuscript design table; compare aggregate_results/sample_counts.json
- Aggregate inputs: sample_counts.json

## tab:regfull — Regression single-encoder results. Houses and horses report $R^2$; facial age reports MAE in years, where lower is better.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 47
- Generator: analysis/generate_manuscript_tables.py
- Aggregate inputs: fixed_all_methods.csv

## tab:classfull — Classification single-encoder accuracy. Selection uses validation log loss, not the printed test accuracy.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 55
- Generator: analysis/generate_manuscript_tables.py
- Aggregate inputs: fixed_all_methods.csv

## tab:combofull — Fixed-split combination candidates. Each candidate is fitted without outer-validation outcomes. Age is MAE; other columns are $R^2$ or accuracy.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 63
- Generator: analysis/generate_manuscript_tables.py
- Aggregate inputs: fixed_all_methods.csv

## fig:appcorrelations — Prediction and residual correlation diagnostics for houses (top) and horses (bottom). Each matrix uses the corrected fitted single-encoder predictions on the same locked observations.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 80
- Generator: analysis/generate_manuscript_figures.py
- Aggregate inputs: correlations.json

## tab:logme — LogME rank association with validation ranking. Winner agreement is a screening diagnostic and never replaces validation.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 98
- Generator: analysis/generate_manuscript_tables.py
- Aggregate inputs: diagnostics/logme_rank_diagnostics.csv

## tab:nnmotivation — Linear and neural heads for the two motivating applications. Values are locked-test $R^2$; neural metrics use predictions averaged across five seeds fixed in the implemented workflow.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 122
- Generator: analysis/generate_neural_assets.py
- Aggregate inputs: fixed_all_methods.csv, neural_head_metrics.csv

## tab:agenn — Continuous-age neural-head MAE and across-seed SD, both in years.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 130
- Generator: analysis/generate_neural_assets.py
- Aggregate inputs: neural_head_metrics.csv

## fig:appnn — Neural-head robustness. Left: within-task normalized neural-head improvement for every dataset--encoder cell; positive values favor the neural head. Right: mean linear-head rank against mean neural-head rank across the six tasks. Ties receive average ranks and lower is better.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/appendix.tex, line 138
- Generator: analysis/generate_neural_assets.py
- Aggregate inputs: fixed_all_methods.csv, neural_head_metrics.csv

## tab:cost — Recorded encoder characteristics and descriptive median cache-creation throughput across seven task records. These heterogeneous timings are not a standardized comparison on common hardware and a common extraction protocol.

Recorded values retained. Seven heterogeneous task records qualified under C05A; no controlled speed or deployment-latency comparison.

- Source: appendix/appendix.tex, line 155
- Generator: analysis/generate_cost_table.py
- Aggregate inputs: cost_records.json

## fig:flowers-selection — Outer-validation log loss for ten frozen representations under training-only tuning. Lower is better; the highlighted representation was selected without using test outcomes.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/public_flowers_demo.tex, line 20
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: fixed_metrics.csv

## fig:flowers-uncertainty — All-ten stack minus validation-selected SigLIP~2-B/16 on the locked test. Bars are paired 95\% conditional bootstrap intervals from fixed fitted predictions. Zero denotes no difference; negative log-loss differences favor the stack.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/public_flowers_demo.tex, line 35
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: stack_vs_selected_paired_bootstrap.json

## fig:flowers-stability — Empirical test-accuracy variation across twelve group-safe splits fixed in the implemented workflow for the selected single encoder and all-ten cross-fitted stack. Bars show empirical 2.5th--97.5th percentiles, not confidence intervals.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/public_flowers_demo.tex, line 50
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: repeated_metrics.csv

## fig:flowers-cost — Observed representation extraction time and outer-validation log loss on the recorded analysis environment. Timings describe this hardware and sample and should not be generalized mechanically.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/public_flowers_demo.tex, line 63
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: fixed_metrics.csv, extraction_costs.json

## tab:flowers-performance — Locked-test Flowers performance after training-only tuning. Selection uses outer-validation log loss.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/tables/encoder_performance.tex, line 1
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: fixed_metrics.csv

## tab:flowers-stability — Flowers split stability under nested out-of-fold fitting and outer-validation selection.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/tables/split_stability.tex, line 1
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: repeated_metrics.csv

## tab:flowers-uncertainty — Paired Flowers comparison from fixed fitted test predictions. Differences are stack minus selected single.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/tables/paired_uncertainty.tex, line 1
- Generator: analysis/generate_flowers_artifacts.py
- Aggregate inputs: stack_vs_selected_paired_bootstrap.json

## tab:flowers-design — Public TensorFlow Flowers demonstration: sample construction and locked design.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/tables/study_design.tex, line 1
- Generator: analysis/generate_design_cost.py
- Aggregate inputs: preparation_summary.json, repeated_metrics.csv

## tab:flowers-cost — Fresh encoder extraction on the replication machine (RTX 3060, 3,668 images). Timings exclude model download.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: appendix/flowers_public_demo/tables/computational_cost.tex, line 1
- Generator: analysis/generate_design_cost.py
- Aggregate inputs: extraction_costs.json

## tab:datasets — Main datasets and locked test designs. $N$ is the common-complete sample used for encoder comparisons.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: main.tex, line 80
- Generator: Inline manuscript design table; compare aggregate_results/sample_counts.json
- Aggregate inputs: sample_counts.json

## fig:motivation — Representation choice and combination in the two motivating applications. Dots report locked-test $R^2$; horizontal bars are 95\% grouped-bootstrap intervals from fixed fitted predictions. The same observations, dimension control, and ridge-head protocol are used within each panel. The final row is the procedure selected on outer-validation loss: a linear stack for houses and feature union for horses.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: main.tex, line 111
- Generator: analysis/generate_manuscript_figures.py
- Aggregate inputs: fixed_headline.csv, figure_intervals.json

## tab:headline — Locked-test headline results. The selected single minimizes outer-validation loss over all ten encoders. The preferred procedure minimizes comparable outer-validation loss over that single and the combination menu fixed in the implemented workflow. $$ is oriented so positive values favor the preferred procedure.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: main.tex, line 120
- Generator: analysis/generate_manuscript_tables.py
- Aggregate inputs: fixed_headline.csv

## fig:crossdomain — Encoder choice and preferred procedure across the six main tasks. Houses and horses report $R^2$; BreaKHis, pneumonia, and rice report accuracy. Facial age is shown on its own MAE-in-years scale, where lower is better. Selection uses validation MSE, MAE, or log loss as appropriate, not the displayed test metric.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: main.tex, line 129
- Generator: analysis/generate_manuscript_figures.py
- Aggregate inputs: fixed_headline.csv

## tab:repeated — Twelve group-safe repeated splits. Age columns are MAE; other performance columns use $R^2$ or accuracy. Gains are oriented so positive values favor the preferred procedure: preferred minus selected-single performance for $R^2$ and accuracy, and selected-single minus preferred MAE for age. ``Preferred SD'' is the sample standard deviation of preferred-procedure test performance across the twelve partitions, not a standard error. Means and differences are computed before rounding. ``Most selected'' reports all encoders tied at the maximum frequency.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: main.tex, line 150
- Generator: analysis/generate_manuscript_tables.py
- Aggregate inputs: repeated_summary.csv

## fig:nn — Flexible heads do not eliminate the ResNet50--DINOv2 gap. Dots report locked-test $R^2$ for the common linear head and the small neural head on houses and horses. Neural metrics use predictions averaged across five seeds fixed in the implemented workflow. The full encoder comparison and the mean-rank diagnostic, which uses average ranks for ties and lower-is-better ranks, appear in Appendix~app:nnfull.

Displayed content agrees with saved evidence at printed precision; scope limited to this artifact, not a universal historical-reload certificate.

- Source: main.tex, line 175
- Generator: analysis/generate_neural_assets.py
- Aggregate inputs: fixed_all_methods.csv, neural_head_metrics.csv

