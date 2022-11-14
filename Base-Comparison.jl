import Pkg
Pkg.activate .
import FairnessUtils

# Define common problem parameters 
G = 2 
B = 50
n_trials = 20
max_movement = 0.5
reduction = 0.5
window_size = Int(round(B/2))
time_limit = 60.0*6.0

results_base = zeros((n_trials, 8));
results_opt = zeros((n_trials, 8));

results_base_test = zeros((n_trials, 8));
results_opt_test = zeros((n_trials, 8));

# Define task name and path. Expect the path to contain "{task}_train.csv", "{task}_bin_midpoints.csv", "{task}_test.csv" 
# Require that train and test file contains 4 columns --> | score [0,1] | label {0,1} | group {1,2} | bin {1,2,...,B}|
# Require that midpoints file contains a single column ("bin_midpoints") with B scalar values representing the bin midpoints (midpoint between quantiles)
task_name = "acs_west_travel"
data_path = "..."*task_name

for i in 1:n_trials
    folder = "Trial_"*string(i-1)*"/"
    path = data_path*"/"*folder*task_name
    # Get base fairness results
    x_eye = zeros((G,B,B))
    x_eye[1,:,:] = I(B)
    x_eye[2,:,:] = I(B)

    df = DataFrame(CSV.File(path*"_train.csv"));
    midpoints = DataFrame(CSV.File(path*"_bin_midpoints.csv"));
    bin_midpoints = midpoints["bin_midpoints"];
    S, N, Y, N_tot, Y_tot = get_parameters(df, G, B, true, 100);
    scoreObj, auc_base, dpViol_base, dpViolTot_base, eoddsViol_base, eoddsViolTot_base, prpViol_base, prpViolTot_base = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_base[i,:] = [scoreObj, auc_base, dpViol_base, dpViolTot_base, eoddsViol_base, eoddsViolTot_base, prpViol_base, prpViolTot_base];

    eps_dp = min(1, dpViol_base)*reduction
    eps_eodds = min(1, eoddsViol_base)*reduction
    eps_prp = min(1, prpViol_base)*reduction
    
    x_opt_bin, m_bin = solve_multifairness(G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, eps_dp, eps_eodds, eps_prp, max_movement, window_size, time_limit);
    scoreObj, auc_opt, dpViol_opt, dpViolTot_opt, eoddsViol_opt, eoddsViolTot_opt, prpViol_opt, prpViolTot_opt  = get_fairness(x_opt_bin, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_opt[i,:] = [scoreObj, auc_opt, dpViol_opt, dpViolTot_opt, eoddsViol_opt, eoddsViolTot_opt, prpViol_opt, prpViolTot_opt];

    df = DataFrame(CSV.File(path*"_test.csv"));
    S, N, Y, N_tot, Y_tot = get_parameters(df, G, B, true, 100);
    scoreObj, auc_base, dpViol_base, dpViolTot_base, eoddsViol_base, eoddsViolTot_base, prpViol_base, prpViolTot_base = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_base_test[i,:] = [scoreObj, auc_base, dpViol_base, dpViolTot_base, eoddsViol_base, eoddsViolTot_base, prpViol_base, prpViolTot_base];

       # Use randomized binning
#         df = DataFrame(CSV.File(path*"_test.csv"));
#         dfr = apply_random_transitions(df, x_opt_bin, i*5);
#         S, N, Y, N_tot, Y_tot = get_parameters(dfr, G, B, true, 100);
#         scoreObj, auc_opt, dpViol_opt, dpViolTot_opt, eoddsViol_opt, eoddsViolTot_opt, prpViol_opt, prpViolTot_opt = get_fairness(x_eye, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    
    # Use "expected" binning
    scoreObj, auc_opt, dpViol_opt, dpViolTot_opt, eoddsViol_opt, eoddsViolTot_opt, prpViol_opt, prpViolTot_opt = get_fairness(x_opt_bin, G, B, bin_midpoints, S, N, Y, N_tot, Y_tot, false)
    results_opt_test[i,:] = [scoreObj, auc_opt, dpViol_opt, dpViolTot_opt, eoddsViol_opt, eoddsViolTot_opt, prpViol_opt, prpViolTot_opt];
end

# Save results 
results_base_df = DataFrame(results_base, [:scoreObj, :auc, :dpViol, :dpViolTot, :eoddsViol, :eoddsViolTot, :prpViol, :prpViolTot])
CSV.write(data_path*"/train_base.csv", results_base_df)
results_opt_df = DataFrame(results_opt, [:scoreObj, :auc, :dpViol, :dpViolTot, :eoddsViol, :eoddsViolTot, :prpViol, :prpViolTot])
CSV.write(data_path*"/train_opt.csv", results_opt_df)

results_base_df = DataFrame(results_base_test, [:scoreObj, :auc, :dpViol, :dpViolTot, :eoddsViol, :eoddsViolTot, :prpViol, :prpViolTot])
CSV.write(data_path*"/test_base.csv", results_base_df)
results_opt_df = DataFrame(results_opt_test, [:scoreObj, :auc, :dpViol, :dpViolTot, :eoddsViol, :eoddsViolTot, :prpViol, :prpViolTot])
CSV.write(data_path*"/test_opt.csv", results_opt_df)