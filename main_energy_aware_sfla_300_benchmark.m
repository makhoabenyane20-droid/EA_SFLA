%% ============================================================================
%  main_energy_aware_sfla_jsp20262v61.m  (V2 + LIVE PROGRESS)
%  ---------------------------------------------------------------------------
%  HYBRID ENERGY-AWARE SFLA FOR TA01 (15x15) JOB SHOP SCHEDULING
%
%  Objectives:
%    1) Minimize Makespan (Cmax)
%    2) Minimize Total Energy (kWh)
%
%  V2 Improvements:
%    - Two-phase optimization:
%        Phase 1: Makespan-first (lexicographic: Cmax then Energy)
%        Phase 2: Pareto within a Cmax band around bestCmaxEver
%    - Stronger TS neighborhood: all adjacent pairs in critical blocks
%    - Swap in OS representation at exact op positions (stronger move)
%    - Deep TS intensification on stall before restart


function [best_solution, archive, runtime_stats] = main_energy_aware_sfla_300_benchmark(varargin)
    % BENCHMARK_ENERGY_AWARE_SFLA - Core algorithm for benchmarking
    %
    % Inputs (optional):
    %   problem - Structure with problem definition (optional)
    %   opts    - Structure with algorithm parameters (optional)
    %
    % Outputs:
    %   best_solution  - Best solution found (job sequence)
    %   archive        - Pareto archive (fitness and solutions)
    %   runtime_stats  - Timing and iteration statistics
    
    % Handle input arguments
    eval_counter = struct();
    eval_counter.count = 0;

    if nargin >= 1
    problem = varargin{1};
    else
    problem = initialize_taillard_problem('TA01');
    end

    if nargin >= 2
    opts = varargin{2};
    else
    opts = struct();  % Initialize opts as empty struct
    end

    % Now set max_evaluations - check if it exists in opts first
    if isfield(opts, 'max_evaluations')
    eval_counter.max_evaluations = opts.max_evaluations;
    else
    % Set a default value
    eval_counter.max_evaluations = 20000;  % or any reasonable default
    end
    
    t_start = tic;
    
    %% ========================= OPTIONS =========================
    % Set default options if not provided
    default_opts = {
        'population_size', 50;
        'memeplex_count', 2;
        'max_iterations',  40;
        'local_search_iterations', 5;
        'print_interval', 1;
        'archive_enabled', true;
        'archive_filename', fullfile(userpath, 'results', 'pareto_front_rank1.csv');
        'stop_when_reached', false;
        'phase1_ratio', 0.70;
        'cmax_band', 12;
        'hybrid_enable', true;
        'hybrid_best_iters', 80;
        'hybrid_memeplex_prob', 0.40;
        'tabu_tenure', 11;
        'ts_max_moves_per_iter', 50;
        'global_intensify_every', 6;
        'global_intensify_topk', 3;
        'global_intensify_iters', 160;
        'stall_window', 30;
        'restart_fraction', 0.61;
        'deep_intensify_iters', 450;
        'deep_tabu_tenure', 18;
        'verbose', true;
        'base_seed', 42;  % For reproducibility
        'max_evaluations', 2e6;
    };
    
    for i = 1:size(default_opts, 1)
        if ~isfield(opts, default_opts{i, 1})
            opts.(default_opts{i, 1}) = default_opts{i, 2};
        end
    end
    
    % Set random seed for reproducibility
    rng(opts.base_seed, 'twister');
    opts.run_id = 1;
    rng(opts.base_seed + opts.run_id);
    
    % Initialize runtime statistics
    runtime_stats = struct();
    runtime_stats.bestCmax = inf;
    runtime_stats.bestEnergy = inf;
    runtime_stats.evaluation_count = 0;
    runtime_stats.eval_times = [];
    runtime_stats.ls_times = [];
    runtime_stats.iter_times = [];
    runtime_stats.hv_history = nan(opts.max_iterations, 1);
    runtime_stats.archive_sizes = nan(opts.max_iterations, 1);
    runtime_stats.best_cmax_history = nan(opts.max_iterations, 1);
    runtime_stats.leader_cmax_history = nan(opts.max_iterations, 1);
    runtime_stats.leader_energy_history = nan(opts.max_iterations, 1);
    runtime_stats.iteration_modes = cell(opts.max_iterations, 1);
    
    results_dir = fullfile(userpath, 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    if opts.verbose
        fprintf('=== HYBRID ENERGY-AWARE SFLA BENCHMARK ===\n');
        fprintf('Objectives:\n  1) Minimize Makespan (Cmax)\n  2) Minimize Total Energy (kWh)\n\n');
        fprintf('Starting Hybrid Energy-Aware SFLA Optimization (Two-phase) ...\n');
    end
    
    %% Initialization
    population = enhanced_initialization(opts.population_size, problem);
    
    %% Archive
    archive.solutions = {};
    archive.fitness = [];
    
    bestCmaxEver = inf;
    bestSeqEver = [];
    stall = 0;
    
    phase1_iters = max(1, round(opts.phase1_ratio * opts.max_iterations));
    
    %% Main optimization loop
    for iter = 1:opts.max_iterations
        if eval_counter.count >= opts.max_evaluations
            if opts.verbose
            fprintf('Evaluation budget exhausted (%d evaluations).\n', eval_counter.count);
            end
            break;
        end
        
        t_iter_start = tic;
        
        % Decide mode
        if iter <= phase1_iters && bestCmaxEver > opts.stop_when_reached
            mode = "PHASE1_CMAX_FIRST";
        else
            mode = "PHASE2_PARETO_BAND";
        end
        runtime_stats.iteration_modes{iter} = mode;
        
        % 1) Evaluate population
        t_eval = tic;
        [fitness_all, dec_cache] = evaluate_population_fast(population, problem, eval_counter);
        runtime_stats.eval_times(end+1) = toc(t_eval);
        runtime_stats.evaluation_count = eval_counter.count; 
        
               
        % 2) Ranking / sorting
        if mode == "PHASE1_CMAX_FIRST"
            [population, fitness_all, dec_cache] = sort_population_lexi(population, fitness_all, dec_cache);
            rank_all = ones(size(fitness_all, 1), 1);
            crowd_all = zeros(size(fitness_all, 1), 1);
        else
            cmax_limit = bestCmaxEver + opts.cmax_band;
            [rank_all, crowd_all, ~] = pareto_rank_and_crowding_band(fitness_all, cmax_limit);
            [population, ~, ~, ~, dec_cache] = shuffle_by_pareto_with_cache(...
                population, fitness_all, rank_all, crowd_all, dec_cache);
        end
        
        % 3) Partition into memeplexes
        memeplexes = partition_memeplexes_matrix(population, opts.memeplex_count);
        cacheplex = partition_memeplexes_matrix(dec_cache, opts.memeplex_count);
        
        % 4) Memeplex local search
        t_ls = tic;
        for m = 1:opts.memeplex_count
            if mode == "PHASE1_CMAX_FIRST"
                [memeplexes{m}, cacheplex{m}] = hybrid_local_search_fast_mode(...
                    memeplexes{m}, cacheplex{m}, problem, opts, "CMAX");
            else
                cmax_limit = bestCmaxEver + opts.cmax_band;
                [memeplexes{m}, cacheplex{m}] = hybrid_local_search_fast_mode(...
                    memeplexes{m}, cacheplex{m}, problem, opts, "PARETO_BAND", cmax_limit);
            end
        end
        runtime_stats.ls_times(end+1) = toc(t_ls);
        
        % 5) Recombine and re-evaluate
        population = recombine_memeplexes_matrix(memeplexes);
        [fitness_all, dec_cache] = evaluate_population_fast(population, problem, eval_counter);
        
        % 6) Update archive and calculate HV
        if opts.archive_enabled
            [rank_true, ~, ~] = pareto_rank_and_crowding(fitness_all);
            archive = update_archive_with_front(population, fitness_all, rank_true, archive);
            
            % Calculate hypervolume
            F_all = [fitness_all; archive.fitness];
            if ~isempty(F_all)
                ref1 = max(F_all(:, 1)) * 1.10;
                ref2 = max(F_all(:, 2)) * 1.10;
                A_nd = extract_rank1_front(archive.fitness);
                runtime_stats.hv_history(iter) = hypervolume_2d(A_nd, [ref1, ref2]);
                runtime_stats.archive_sizes(iter) = size(A_nd, 1);
            else
                runtime_stats.hv_history(iter) = 0;
                runtime_stats.archive_sizes(iter) = 0;
            end
        end
        
        % 7) Global intensification (rare)
        if opts.hybrid_enable && mod(iter, opts.global_intensify_every) == 0
            [population, fitness_all] = global_intensify(population, fitness_all, problem, opts);
            [fitness_all, dec_cache] = evaluate_population_fast(population, problem, eval_counter);
        end
        
        % 8) Pick leader
        if mode == "PHASE1_CMAX_FIRST"
            [~, leader_idx] = minrows(fitness_all);
        else
            [rank_all2, ~, ~] = pareto_rank_and_crowding(fitness_all);
            r1 = find(rank_all2 == 1);
            if isempty(r1)
                [~, leader_idx] = minrows(fitness_all);
            else
                [~, k] = min(fitness_all(r1, 1));
                leader_idx = r1(k);
            end
        end
        
        leader_fit = fitness_all(leader_idx, :);
        
        % Store leader metrics
        runtime_stats.leader_cmax_history(iter) = leader_fit(1);
        runtime_stats.leader_energy_history(iter) = leader_fit(2);
        
        % Update best solution
        if leader_fit(1) < bestCmaxEver
            bestCmaxEver = leader_fit(1);
            bestSeqEver = population{leader_idx};
            stall = 0;
        else
            stall = stall + 1;
        end
        runtime_stats.best_cmax_history(iter) = bestCmaxEver;
        
        % 9) Stall handling
        if stall >= opts.stall_window && ~isempty(bestSeqEver)
            if opts.verbose
                fprintf('  Stall (%d iters). Deep intensify best-so-far then restart worst %.0f%%...\n', ...
                    opts.stall_window, 100 * opts.restart_fraction);
            end
            
            bestSeqEver = critical_block_tabu_search_swap(bestSeqEver, problem, ...
                opts.deep_intensify_iters, opts.deep_tabu_tenure, opts.stop_when_reached, opts);
            bf = evaluate_schedule_fast(bestSeqEver, problem, eval_counter);
            bestCmaxEver = min(bestCmaxEver, bf(1));
            
            population = restart_population(population, bestSeqEver, problem, opts);
            stall = 0;
        end
        
        % Store iteration time
        runtime_stats.iter_times(end+1) = toc(t_iter_start);
        
        % Progress output
        if opts.verbose && mod(iter, opts.print_interval) == 0
            fprintf('Iter %3d [%s]: Leader Cmax=%.0f, E=%.3f kWh | bestCmax=%.0f\n', ...
                iter, mode, leader_fit(1), leader_fit(2), bestCmaxEver);
        end
        
        % Early stopping
        if opts.stop_when_reached && bestCmaxEver <= opts.stop_when_reached
            if opts.verbose
                fprintf('Target reached: bestCmax=%.0f (<= %d). Stopping.\n', bestCmaxEver, opts.stop_when_reached);
            end
            
        end
    end
    
    %% Final results
    if eval_counter.count < opts.max_evaluations
        [fitness_all_final, ~] = evaluate_population_fast(population, problem, eval_counter);
        else
        fitness_all_final = runtime_stats.final_population_fitness;
    end
    
    % Final statistics
    runtime_stats.total_time = toc(t_start);
    runtime_stats.iterations_completed = iter;
    runtime_stats.final_population_fitness = fitness_all;
    runtime_stats.final_population_size = numel(population);
    
    % Trim history arrays to actual iterations completed
    runtime_stats.hv_history = runtime_stats.hv_history(1:iter);
    runtime_stats.archive_sizes = runtime_stats.archive_sizes(1:iter);
    runtime_stats.best_cmax_history = runtime_stats.best_cmax_history(1:iter);
    runtime_stats.leader_cmax_history = runtime_stats.leader_cmax_history(1:iter);
    runtime_stats.leader_energy_history = runtime_stats.leader_energy_history(1:iter);
    runtime_stats.iteration_modes = runtime_stats.iteration_modes(1:iter);
    
    % Optional export
    if opts.archive_enabled ...
        && ischar(opts.archive_filename) ...
        && ~isempty(opts.archive_filename) ...
        && ~isempty(archive.fitness)

       export_pareto_csv(archive, opts.archive_filename);
    end
    
    % Final output
    best_solution = bestSeqEver;
    if opts.verbose
        display_final_schedule(best_solution, problem);
        fprintf('\n=== BENCHMARK STATISTICS ===\n');
        fprintf('Total runtime: %.2f seconds\n', runtime_stats.total_time);
        fprintf('Iterations completed: %d\n', runtime_stats.iterations_completed);
        fprintf('Average iteration time: %.3f seconds\n', mean(runtime_stats.iter_times));
        if opts.archive_enabled && ~isempty(archive.fitness)
            fprintf('Final archive size: %d solutions\n', size(archive.fitness, 1));
            fprintf('Final hypervolume: %.4f\n', runtime_stats.hv_history(end));
        end
    end
    fprintf('Total evaluations used: %d\n', eval_counter.count);
    
    % =======================
    % PLOTS
    % =======================
    plot_hypervolume(runtime_stats, opts);

    if opts.archive_enabled && ~isempty(archive.fitness)
       plot_pareto_front(archive, runtime_stats, problem);
    end

    plot_convergence(runtime_stats, opts);
    plot_gantt_chart(best_solution, problem, true);
    plot_energy_breakdown(best_solution, problem);
end

function [val, idx] = minrows(F)
    % lexicographic min by col1 then col2 (approx)
    [~, idx] = min(F(:,1) + 1e-6*F(:,2));
    val = F(idx,:);
end

%% ============================================================================
%  PROBLEM INITIALIZATION (TA01 15x15)
% ============================================================================
function problem = initialize_taillard_problem(instance_name)
    % Initialize based on instance name
    switch instance_name
        case 'TA01'
            problem.num_jobs = 15;
            problem.num_machines = 15;
            problem.processing_times = [
                94 66 10 53 26 15 65 82 10 27 93 92 96 70 83;
                74 31 88 51 57 78  8  7 91 79 18 51 18 99 33;
                 4 82 40 86 50 54 21  6 54 68 82 20 39 35 68;
                73 23 30 30 53 94 58 93 32 91 30 56 27 92  9;
                78 23 21 60 36 29 95 99 79 76 93 42 52 42 96;
                29 61 88 70 16 31 65 83 78 26 50 87 62 14 30;
                18 75 20  4 91 68 19 54 85 73 43 24 37 87 66;
                32 52  9 49 61 35 99 62  6 62  7 80  3 57  7;
                85 30 96 91 13 87 82 83 78 56 85  8 66 88 15;
                 5 59 30 60 41 17 66 89 78 88 69 45 82  6 13;
                90 27  1  8 91 80 89 49 32 28 90 93  6 35 73;
                47 43 75  8 51  3 84 34 28 60 69 45 67 58 87;
                65 62 97 20 31 33 33 77 50 80 48 90 75 96 44;
                28 21 51 75 17 89 59 56 63 18 17 30 16  7 35;
                57 16 42 34 37 26 68 73  5  8 12 87 83 20 97;
            ];
            problem.machine_sequences = [
                 7 13  5  8  4  3 11 12  9 15 10 14  6  1  2;
                 5  6  8 15 14  9 12 10  7 11  1  4 13  2  3;
                 2  9 10 13  7 12 14  6  1  3  8 11  5  4 15;
                 6  3 10  7 11  1 14  5  8 15 12  9 13  2  4;
                 8  9  7 11  5 10  3 15 13  6  2 14 12  1  4;
                 6  4 13 14 12  5 15  8  3  2 11  1 10  7  9;
                13  4  8  9 15  7  2 12  5  6  3 11  1 14 10;
                12  6  1  8 13 14 15  2  3  9  5  4 10  7 11;
                11 12  7 15  1  2  3  6 13  5  9  8 10 14  4;
                 7 12 10  3  9  1 14  4 11  8  2 13 15  5  6;
                 5  8 14  1  6 13  7  9 15 11  4  2 12 10  3;
                 3 15  1 13  7 11  8  6  9 10 14  2  4 12  5;
                 6  9 11  3  4  7 10  1 14  5  2 12 13  8 15;
                 9 15  5 14  6  7 10  2 13  8 12 11  4  3  1;
                11  9 13  7  5  2 14 15 12  1  8  4  3 10  6;
            ];
            
        case 'TA02'
            % Add TA02 data here (20x20 instance)
            problem.num_jobs = 20;
            problem.num_machines = 20;
            % Add processing times and machine sequences for TA02
            % ... (load from file or hardcode)
            
        case 'TA03'
            % Add TA03 data here (20x20 instance)
            % ... (load from file or hardcode)
            
        otherwise
            error('Instance %s not supported', instance_name);
    end

    problem.time_unit_hours = 1/60;

    problem.energy.sleep_threshold_min = 15;
    problem.energy.wakeup_time_min    = 2.5;

    problem.energy.sleep_power   = [0.15, 0.12, 0.13, 0.15, 0.14, 0.15, 0.14, 0.12, 0.17, 0.17, 0.17, 0.13, 0.15, 0.13, 0.19];
    problem.energy.wakeup_energy = [0.08, 0.06, 0.07, 0.08, 0.07, 0.08, 0.07, 0.06, 0.09, 0.09, 0.09, 0.07, 0.08, 0.07, 0.10];
    
    % Detailed power consumption parameters 
    problem.idle_power = [0.8, 0.7, 0.75, 0.85, 0.9, 0.8, 0.75, 0.7, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6];
    problem.runtime_power = [2.5, 2.3, 2.4, 2.8, 2.2, 2.6, 2.4, 2.1, 2.7, 2.9, 2.8, 2.2, 2.6, 2.3, 3.0];
    problem.energy.wakeup_power = [1.8, 1.6, 1.7, 1.9, 1.6, 1.8, 1.7, 1.6, 1.9, 2.0, 1.9, 1.6, 1.8, 1.7, 2.1];

    assert(numel(problem.energy.sleep_power)==problem.num_machines,'sleep_power length mismatch');
    assert(numel(problem.energy.wakeup_energy)==problem.num_machines,'wakeup_energy length mismatch');
    
    % Validate power parameter dimensions
    assert(numel(problem.idle_power) == problem.num_machines, ...
        'idle_power must have %d elements (one per machine)', problem.num_machines);
    assert(numel(problem.runtime_power) == problem.num_machines, ...
        'runtime_power must have %d elements (one per machine)', problem.num_machines);
    assert(numel(problem.energy.wakeup_power) == problem.num_machines, ...
        'wakeup_power must have %d elements (one per machine)', problem.num_machines);
    
    % Validate power values are positive
    assert(all(problem.idle_power >= 0), 'idle_power must be non-negative');
    assert(all(problem.runtime_power >= 0), 'runtime_power must be non-negative');
    assert(all(problem.energy.wakeup_power >= 0), 'wakeup_power must be non-negative');
    
    % Validate that runtime power is greater than idle power (logical check)
    for m = 1:problem.num_machines
        if problem.runtime_power(m) <= problem.idle_power(m)
            warning('Machine %d: runtime_power (%.2f) <= idle_power (%.2f). This may not be physically realistic.', ...
                m, problem.runtime_power(m), problem.idle_power(m));
        end
    end
end


%% ============================================================================
%  INITIAL POPULATION (heuristic seeding)
% ============================================================================
function population = enhanced_initialization(population_size, problem)
    population = cell(population_size, 1);

    num_spt    = round(population_size * 0.30);
    num_mwr    = round(population_size * 0.30);
    num_lpt    = round(population_size * 0.20);
    num_random = population_size - num_spt - num_mwr - num_lpt;

    idx = 1;
    for i = 1:num_spt
        population{idx} = build_greedy_sequence(problem, 'spt'); idx = idx + 1;
    end
    for i = 1:num_mwr
        population{idx} = build_greedy_sequence(problem, 'mwr'); idx = idx + 1;
    end
    for i = 1:num_lpt
        population{idx} = build_greedy_sequence(problem, 'lpt'); idx = idx + 1;
    end

    for i = 1:num_random
        base_seq = [];
        for job = 1:problem.num_jobs
            base_seq = [base_seq, repmat(job, 1, problem.num_machines)]; %#ok<AGROW>
        end
        population{idx} = base_seq(randperm(numel(base_seq)));
        idx = idx + 1;
    end
end

function seq = build_greedy_sequence(problem, mode)
    nj = problem.num_jobs; nm = problem.num_machines;

    job_prog   = ones(nj,1);
    job_ready  = zeros(nj,1);
    mach_ready = zeros(nm,1);

    seq = zeros(1, nj*nm);

    for k = 1:(nj*nm)
        cand = find(job_prog <= nm);
        scores = inf(size(cand));
        starts = inf(size(cand));

        for ii = 1:numel(cand)
            j  = cand(ii);
            op = job_prog(j);
            m  = problem.machine_sequences(j, op);
            pt = problem.processing_times(j, m);

            rt = max(job_ready(j), mach_ready(m));
            starts(ii) = rt;

            switch lower(mode)
                case 'spt'
                    scores(ii) = pt;
                case 'lpt'
                    scores(ii) = -pt;
                case 'mwr'
                    rem = 0;
                    for op2 = op:nm
                        m2 = problem.machine_sequences(j, op2);
                        rem = rem + problem.processing_times(j, m2);
                    end
                    scores(ii) = -rem;
                otherwise
                    scores(ii) = pt;
            end
        end

        [~, besti] = min(scores + 1e-3*starts);
        jbest = cand(besti);

        op = job_prog(jbest);
        m  = problem.machine_sequences(jbest, op);
        pt = problem.processing_times(jbest, m);

        rt = max(job_ready(jbest), mach_ready(m));
        ct = rt + pt;

        job_ready(jbest) = ct;
        mach_ready(m)    = ct;
        job_prog(jbest)  = job_prog(jbest) + 1;

        seq(k) = jbest;
    end
end

%% ============================================================================
%  SORTING HELPERS
% ============================================================================
function [population, fitness, cache] = sort_population_lexi(population, fitness, cache)
    [~, idx] = sortrows(fitness, [1 2]);
    population = population(idx);
    fitness    = fitness(idx,:);
    cache      = cache(idx);
end

%% ============================================================================
%  FAST EVALUATION (decode once; energy from machine intervals)
% ============================================================================
function [fitness, dec_cache] = evaluate_population_fast(population, problem, eval_counter)
    N = numel(population);
    fitness = zeros(N, 2);
    dec_cache = cell(N, 1);
    
    for i = 1:N
        [fitness(i, :), dec_cache{i}] = evaluate_schedule_fast(population{i}, problem, eval_counter);
    end
end

function [fitness, dec] = evaluate_schedule_fast(job_sequence, problem, eval_counter)

    if nargin >= 3 && eval_counter.count >= eval_counter.max_evaluations
        fitness = [inf inf];
        dec = [];
        return;
    end

    if nargin >= 3
        eval_counter.count = eval_counter.count + 1;
    end
    dec = decode_schedule(job_sequence, problem);
    makespan = dec.makespan;
    total_energy = total_energy_from_dec(dec, problem);
    fitness = [makespan, total_energy];
end

function dec = decode_schedule(job_sequence, problem)
    nj = problem.num_jobs;
    nm = problem.num_machines;

    job_progress = ones(nj,1);
    st = nan(nj,nm);
    ct = zeros(nj,nm);
    pos_of_op = zeros(nj,nm);

    mach_ready = zeros(nm,1);
    mach_list  = cell(nm,1); % each: [st, job, op]

    for pos = 1:numel(job_sequence)
        job = job_sequence(pos);
        op  = job_progress(job);
        if op > nm, continue; end

        m  = problem.machine_sequences(job, op);
        pt = problem.processing_times(job, m);

        job_ready = 0;
        if op > 1
            job_ready = ct(job, op-1);
        end

        start_t = max(job_ready, mach_ready(m));
        end_t   = start_t + pt;

        st(job,op) = start_t;
        ct(job,op) = end_t;
        pos_of_op(job,op) = pos;

        mach_ready(m) = end_t;
        mach_list{m}  = [mach_list{m}; start_t, job, op]; 

        job_progress(job) = op + 1;
    end

    mach_order = cell(nm,1);
    mach_iv    = cell(nm,1);
    for m = 1:nm
        A = mach_list{m};
        if isempty(A)
            mach_order{m} = zeros(0,2);
            mach_iv{m}    = zeros(0,2);
        else
            A = sortrows(A,1);
            mach_order{m} = A(:,2:3);

            stv = A(:,1);
            endv = zeros(size(stv));
            for k = 1:numel(stv)
                j = A(k,2);
                endv(k) = stv(k) + problem.processing_times(j, m);
            end
            mach_iv{m} = [stv, endv];
        end
    end

    dec.st = st;
    dec.ct = ct;
    dec.pos_of_op = pos_of_op;
    dec.mach_order = mach_order;
    dec.mach_iv = mach_iv;
    dec.makespan = max(ct(:,end));
end

function totalE = total_energy_from_dec(dec, problem)
    nm = problem.num_machines;
    th_sleep = problem.energy.sleep_threshold_min;
    sleepP   = problem.energy.sleep_power;
    wakeE_k  = problem.energy.wakeup_energy;
    t_wu_min = problem.energy.wakeup_time_min;
    t_wu_h   = t_wu_min * problem.time_unit_hours;

    runtimeE = 0; idleOnE = 0; sleepE = 0; wakeE = 0;

    for m = 1:nm
        iv = dec.mach_iv{m};
        makespan = dec.makespan;

        if isempty(iv)
            gap_h = makespan * problem.time_unit_hours;
            sleepE = sleepE + gap_h * sleepP(m);
            continue;
        end

        runtime_min = sum(iv(:,2) - iv(:,1));
        runtimeE = runtimeE + (runtime_min * problem.time_unit_hours) * problem.runtime_power(m);

        gap = iv(1,1) - 0;
        if gap > 0
            [e_on, e_sl, e_wu] = gap_energy_components_fast(m, gap, th_sleep, sleepP, wakeE_k, t_wu_min, t_wu_h, problem, false);
            idleOnE = idleOnE + e_on; sleepE = sleepE + e_sl; wakeE = wakeE + e_wu;
        end

        for k = 1:size(iv,1)-1
            gap = iv(k+1,1) - iv(k,2);
            if gap > 0
                [e_on, e_sl, e_wu] = gap_energy_components_fast(m, gap, th_sleep, sleepP, wakeE_k, t_wu_min, t_wu_h, problem, false);
                idleOnE = idleOnE + e_on; sleepE = sleepE + e_sl; wakeE = wakeE + e_wu;
            end
        end

        gap = makespan - iv(end,2);
        if gap > 0
            [e_on, e_sl, e_wu] = gap_energy_components_fast(m, gap, th_sleep, sleepP, wakeE_k, t_wu_min, t_wu_h, problem, true);
            idleOnE = idleOnE + e_on; sleepE = sleepE + e_sl; wakeE = wakeE + e_wu;
        end
    end

    totalE = runtimeE + idleOnE + sleepE + wakeE;
end

function [e_on, e_sleep, e_wu] = gap_energy_components_fast(m, gap_min, th_sleep, sleepP, wakeE_k, t_wu_min, t_wu_h, problem, is_tail)
    g_h = gap_min * problem.time_unit_hours;

    if gap_min >= th_sleep
        e_on = 0;
        if ~is_tail
            sleep_min = max(0, gap_min - t_wu_min);
            e_sleep   = (sleep_min * problem.time_unit_hours) * sleepP(m);
            e_wu      = wakeE_k(m) + t_wu_h * problem.energy.wakeup_power(m);
        else
            e_sleep = g_h * sleepP(m);
            e_wu    = 0;
        end
    else
        e_on    = g_h * problem.idle_power(m);
        e_sleep = 0;
        e_wu    = 0;
    end
end

%% ============================================================================
%  PARETO RANKING (standard)
% ============================================================================
function [rank, crowd, fronts] = pareto_rank_and_crowding(fitness)
    N = size(fitness,1);
    S = cell(N,1);
    n = zeros(N,1);
    rank = zeros(N,1);
    fronts = {};

    F1 = [];
    for p = 1:N
        Sp = [];
        np = 0;
        for q = 1:N
            if p==q, continue; end
            if dominates_vec(fitness(p,:), fitness(q,:))
                Sp = [Sp, q]; 
            elseif dominates_vec(fitness(q,:), fitness(p,:))
                np = np + 1;
            end
        end
        S{p} = Sp;
        n(p) = np;
        if n(p)==0
            rank(p)=1;
            F1 = [F1, p];
        end
    end

    fronts{1} = F1;
    i = 1;
    while ~isempty(fronts{i})
        Q = [];
        for p = fronts{i}
            for q = S{p}
                n(q) = n(q) - 1;
                if n(q)==0
                    rank(q)=i+1;
                    Q = [Q, q]; 
                end
            end
        end
        i = i + 1;
        fronts{i} = Q; 
    end
    if isempty(fronts{end}), fronts(end)=[]; end

    crowd = crowding_distance(fitness, fronts);
end

function [rank, crowd, fronts] = pareto_rank_and_crowding_band(fitness, cmax_limit)
    % Any solution with Cmax > cmax_limit is penalized (rank pushed down).
    % Within band => normal Pareto among themselves.
    N = size(fitness,1);
    inBand = fitness(:,1) <= cmax_limit;

    rank  = zeros(N,1);
    crowd = zeros(N,1);
    fronts = {};

    if any(inBand)
        idxBand = find(inBand);
        Fb = fitness(idxBand,:);
        [rb, cb, frontsB] = pareto_rank_and_crowding(Fb);

        rank(idxBand)  = rb;
        crowd(idxBand) = cb;

        % out-of-band = very bad ranks
        maxrb = max(rb);
        out = find(~inBand);
        rank(out) = maxrb + 10;
        crowd(out)= 0;

        fronts = frontsB;
    else
        rank(:) = 1;
        crowd(:)= 0;
        fronts{1} = 1:N;
    end
end

function cd = crowding_distance(fitness, fronts)
    N = size(fitness,1);
    cd = zeros(N,1);
    M = size(fitness,2);

    for f = 1:numel(fronts)
        idx = fronts{f};
        if isempty(idx), continue; end
        Ff = fitness(idx,:);
        K  = numel(idx);
        if K==1
            cd(idx)=inf; continue;
        end

        cdf = zeros(K,1);
        for m = 1:M
            [vals, order] = sort(Ff(:,m), 'ascend');
            cdf(order(1))   = inf;
            cdf(order(end)) = inf;
            denom = max(vals)-min(vals); if denom<eps, denom=1; end
            for k = 2:K-1
                cdf(order(k)) = cdf(order(k)) + (vals(k+1)-vals(k-1))/denom;
            end
        end
        cd(idx)=cdf;
    end
end

function tf = dominates_vec(a, b)
    tf = all(a <= b) && any(a < b);
end

function [population, fitness, rank, crowd, cache] = shuffle_by_pareto_with_cache(population, fitness, rank, crowd, cache)
    [~, idx] = sortrows([rank, -crowd, fitness(:,1), fitness(:,2)]);
    population = population(idx);
    fitness    = fitness(idx,:);
    rank       = rank(idx);
    crowd      = crowd(idx);
    cache      = cache(idx);
end

function memeplexes = partition_memeplexes_matrix(population, memeplex_count)
    N = numel(population);
    memeplexes = cell(memeplex_count, 1);
    for m = 1:memeplex_count
        memeplexes{m} = population(m:memeplex_count:N);
    end
end

function population = recombine_memeplexes_matrix(memeplexes)
    population = {};
    for m = 1:numel(memeplexes)
        population = [population; memeplexes{m}(:)]; %#ok<AGROW>
    end
end

%% ============================================================================
%  HYBRID LOCAL SEARCH (mode-aware)
% ============================================================================
function [memeplex, cacheplex] = hybrid_local_search_fast_mode(memeplex, cacheplex, problem, opts, mode, varargin)
    if isempty(memeplex), return; end
    M = numel(memeplex);

    fitness = zeros(M,2);
    for i = 1:M
        if isempty(cacheplex{i})
            [fitness(i,:), cacheplex{i}] = evaluate_schedule_fast(memeplex{i}, problem);
        else
            dec = cacheplex{i};
            fitness(i,:) = [dec.makespan, total_energy_from_dec(dec, problem)];
        end
    end

    if mode == "CMAX"
        [~, order] = sortrows(fitness,[1 2]);
    else
        cmax_limit = varargin{1};
        [r, c, ~] = pareto_rank_and_crowding_band(fitness, cmax_limit);
        [~, order] = sortrows([r, -c, fitness(:,1), fitness(:,2)]);
    end

    % One TS on memeplex best
    best_idx = order(1);
    if opts.hybrid_enable
        s0 = memeplex{best_idx};
        s1 = critical_block_tabu_search_swap(s0, problem, opts.hybrid_best_iters, opts.tabu_tenure, opts.stop_when_reached, opts);
        [f1, d1] = evaluate_schedule_fast(s1, problem);
        if lex_better(f1, fitness(best_idx,:))
            memeplex{best_idx}  = s1;
            cacheplex{best_idx} = d1;
            fitness(best_idx,:) = f1;
        end
    end

    for it = 1:opts.local_search_iterations 
        if mode == "CMAX"
            [~, order] = sortrows(fitness,[1 2]);
        else
            cmax_limit = varargin{1};
            [r, c, ~] = pareto_rank_and_crowding_band(fitness, cmax_limit);
            [~, order] = sortrows([r, -c, fitness(:,1), fitness(:,2)]);
        end

        for w = 1:ceil(M*0.5)
            worst_idx = order(end-w+1);
            base_seq = memeplex{worst_idx};
            base_fit = fitness(worst_idx,:);

            cand_set = {};
            if opts.hybrid_enable && rand() < opts.hybrid_memeplex_prob
                cand_set{end+1} = critical_block_tabu_search_swap(base_seq, problem, 25, opts.tabu_tenure, opts.stop_when_reached, opts);
            end

            cand_set{end+1} = multi_swap_mutation(base_seq);
            cand_set{end+1} = segment_shuffle_mutation(base_seq);
            cand_set{end+1} = reverse_segment_mutation(base_seq);
            cand_set{end+1} = critical_job_mutation(base_seq, problem);

            bestCand = base_seq;
            bestFit  = base_fit;
            bestDec  = cacheplex{worst_idx};

            for k = 1:numel(cand_set)
                [cf, cd] = evaluate_schedule_fast(cand_set{k}, problem);

                if mode ~= "CMAX"
                    cmax_limit = varargin{1};
                    if cf(1) > cmax_limit
                        continue; % respect band in phase2
                    end
                end

                if lex_better(cf, bestFit)
                    bestFit  = cf;
                    bestCand = cand_set{k};
                    bestDec  = cd;
                end
            end

            if lex_better(bestFit, base_fit)
                memeplex{worst_idx}  = bestCand;
                cacheplex{worst_idx} = bestDec;
                fitness(worst_idx,:) = bestFit;
            end
        end
    end
end

function tf = lex_better(a,b)
    tf = (a(1) < b(1)) || (a(1)==b(1) && a(2) < b(2));
end

%% ============================================================================
%  GLOBAL INTENSIFICATION
% ============================================================================
function [population, fitness_all] = global_intensify(population, fitness_all, problem, opts)
    [~, idx] = sortrows(fitness_all, [1 2]);
    topk = idx(1:min(opts.global_intensify_topk, numel(idx)));

    for t = 1:numel(topk)
        i = topk(t);
        s0 = population{i};
        s1 = critical_block_tabu_search_swap(s0, problem, opts.global_intensify_iters, opts.tabu_tenure, opts.stop_when_reached, opts);
        f1 = evaluate_schedule_fast(s1, problem);
        if lex_better(f1, fitness_all(i,:))
            population{i}  = s1;
            fitness_all(i,:) = f1;
        end
    end
end

%% ============================================================================
%  CRITICAL PATH + CRITICAL BLOCK MOVES + STRONG TABU (SWAP)
% ============================================================================
function crit = critical_path_ops(dec, problem)
    nj = problem.num_jobs; nm = problem.num_machines;
    st = dec.st; ct = dec.ct;
    mach_order = dec.mach_order;

    predM = zeros(nj,nm,2);
    for m = 1:nm
        ord = mach_order{m};
        for k = 2:size(ord,1)
            j2 = ord(k,1); o2 = ord(k,2);
            j1 = ord(k-1,1); o1 = ord(k-1,2);
            predM(j2,o2,1) = j1;
            predM(j2,o2,2) = o1;
        end
    end

    [~, jend] = max(ct(:,end));
    oend = nm;

    crit = false(nj,nm);
    stack = [jend, oend];
    tol = 1e-9;

    while ~isempty(stack)
        node = stack(end,:); stack(end,:) = [];
        j = node(1); o = node(2);
        if j==0 || o==0 || crit(j,o), continue; end
        crit(j,o) = true;

        s = st(j,o);

        if o > 1
            pj = j; po = o-1;
            if abs(ct(pj,po) - s) < tol
                stack = [stack; pj, po]; 
            end
        end

        pmj = predM(j,o,1); pmo = predM(j,o,2);
        if pmj ~= 0
            if abs(ct(pmj,pmo) - s) < tol
                stack = [stack; pmj, pmo]; %#ok<AGROW>
            end
        end
    end
end

function pairs = critical_block_adjacent_pairs(dec, crit, problem)
    nm = problem.num_machines;
    mach_order = dec.mach_order;

    pairs = []; % rows: [jA oA jB oB]
    for m = 1:nm
        ord = mach_order{m};
        if size(ord,1) < 2, continue; end

        isC = false(size(ord,1),1);
        for k = 1:size(ord,1)
            isC(k) = crit(ord(k,1), ord(k,2));
        end

        k = 1;
        while k <= numel(isC)
            if ~isC(k), k = k+1; continue; end
            s = k;
            while k <= numel(isC) && isC(k), k = k+1; end
            e = k-1;

            % collect ALL adjacent pairs in the block
            if e - s + 1 >= 2
                for t = s:(e-1)
                    pairs = [pairs; ord(t,1) ord(t,2) ord(t+1,1) ord(t+1,2)]; 
                end
            end
        end
    end
end

function best_seq = critical_block_tabu_search_swap(seq0, problem, iters, tenure, targetCmax, opts)
    nj = problem.num_jobs; nm = problem.num_machines;

    seq = seq0;
    [best_fit, best_dec] = evaluate_schedule_fast(seq, problem);
    best_seq = seq;

    tabuExpire = zeros(nj,nm,nj,nm,'int32');

    for t = 1:iters
        crit = critical_path_ops(best_dec, problem);
        mv   = critical_block_adjacent_pairs(best_dec, crit, problem);
        if isempty(mv), break; end

        % cap moves for speed
        if size(mv,1) > opts.ts_max_moves_per_iter
            mv = mv(randperm(size(mv,1), opts.ts_max_moves_per_iter), :);
        else
            mv = mv(randperm(size(mv,1)),:);
        end

        bestCandFit = [inf inf];
        bestCandSeq = [];
        bestPair = [0 0 0 0];

        for i = 1:size(mv,1)
            jA = mv(i,1); oA = mv(i,2);
            jB = mv(i,3); oB = mv(i,4);

            pA = best_dec.pos_of_op(jA,oA);
            pB = best_dec.pos_of_op(jB,oB);
            if pA==0 || pB==0 || pA==pB, continue; end

            isTabu = (t < tabuExpire(jA,oA,jB,oB));

            cand = swap_positions(seq, pA, pB);
            candFit = evaluate_schedule_fast(cand, problem);

            if isTabu && candFit(1) >= best_fit(1)
                continue; % tabu unless aspiration by makespan
            end

            if lex_better(candFit, bestCandFit)
                bestCandFit = candFit;
                bestCandSeq = cand;
                bestPair = [jA oA jB oB];
            end
        end

        if isempty(bestCandSeq), break; end

        seq = bestCandSeq;
        [cur_fit, cur_dec] = evaluate_schedule_fast(seq, problem);

        jA=bestPair(1); oA=bestPair(2); jB=bestPair(3); oB=bestPair(4);
        if jA>0
            tabuExpire(jA,oA,jB,oB) = int32(t + tenure);
            tabuExpire(jB,oB,jA,oA) = int32(t + tenure);
        end

        if lex_better(cur_fit, best_fit)
            best_fit = cur_fit;
            best_dec = cur_dec;
            best_seq = seq;

            if best_fit(1) <= targetCmax
                return;
            end
        else
            best_dec = cur_dec;
        end
    end
end

function seq2 = swap_positions(seq, pA, pB)
    seq2 = seq;
    seq2([pA pB]) = seq2([pB pA]);
end

%% ============================================================================
%  MUTATIONS
% ============================================================================
function new_seq = swap_mutation(sequence)
    new_seq = sequence;
    L = numel(new_seq);
    if L < 2, return; end
    i = randi(L); j = randi(L);
    new_seq([i j]) = new_seq([j i]);
end

function new_seq = multi_swap_mutation(sequence)
    new_seq = sequence;
    L = numel(new_seq);
    ns = randi([2,4]);
    for s = 1:ns
        i = randi(L); j = randi(L);
        new_seq([i j]) = new_seq([j i]);
    end
end

function new_seq = segment_shuffle_mutation(sequence)
    L = numel(sequence);
    if L < 4, new_seq = swap_mutation(sequence); return; end
    seglen = randi([3, min(8, L)]);
    i = randi([1, L-seglen+1]);
    new_seq = sequence;
    seg = sequence(i:i+seglen-1);
    new_seq(i:i+seglen-1) = seg(randperm(seglen));
end

function new_seq = reverse_segment_mutation(sequence)
    L = numel(sequence);
    if L < 4, new_seq = swap_mutation(sequence); return; end
    seglen = randi([2, min(10, L)]);
    i = randi([1, L-seglen+1]);
    new_seq = sequence;
    new_seq(i:i+seglen-1) = fliplr(sequence(i:i+seglen-1));
end

function new_seq = critical_job_mutation(sequence, problem)
    [~, dec] = evaluate_schedule_fast(sequence, problem);
    makespan = dec.makespan;
    end_times = dec.ct(:, end);
    critical_jobs = find(end_times >= makespan * 0.9);

    if isempty(critical_jobs)
        new_seq = multi_swap_mutation(sequence);
        return;
    end

    new_seq = sequence;
    for attempt = 1:min(4, numel(critical_jobs))
        job = critical_jobs(attempt);
        pos = find(new_seq==job);
        if numel(pos) > 1
            pick = pos(randi(numel(pos)));
            if pick > 2
                new_pos = randi(pick-1);
                tmp = new_seq;
                tmp(pick) = [];
                tmp = [tmp(1:new_pos-1), job, tmp(new_pos:end)];
                new_seq = tmp;
            end
        end
    end
end

function new_seq = intensive_mutation(sequence, problem)
    new_seq = sequence;
    nmuts = randi([4,7]);
    for k = 1:nmuts
        t = randi(5);
        switch t
            case 1, new_seq = multi_swap_mutation(new_seq);
            case 2, new_seq = segment_shuffle_mutation(new_seq);
            case 3, new_seq = critical_job_mutation(new_seq, problem);
            case 4, new_seq = reverse_segment_mutation(new_seq);
            case 5, new_seq = swap_mutation(new_seq);
        end
    end
end

%% ============================================================================
%  RESTART
% ============================================================================
function population = restart_population(population, elite_seq, problem, opts)
    N = numel(population);
    nrep = max(1, round(N*opts.restart_fraction));
    for i = (N-nrep+1):N
        r = rand();
        if r < 0.55
            population{i} = critical_block_tabu_search_swap(elite_seq, problem, 90, opts.tabu_tenure, opts.stop_when_reached, opts);
        elseif r < 0.85
            population{i} = intensive_mutation(elite_seq, problem);
        else
            base_seq = [];
            for job = 1:problem.num_jobs
                base_seq = [base_seq, repmat(job, 1, problem.num_machines)]; %#ok<AGROW>
            end
            population{i} = base_seq(randperm(numel(base_seq)));
        end
    end
end

%% ============================================================================
%  ARCHIVE + HV
% ============================================================================
function archive = update_archive_with_front(population, fitness_all, rank_all, archive)
    idx = (rank_all==1);
    sols = population(idx);
    fits = fitness_all(idx,:);

    all_sols = [archive.solutions; sols(:)];
    all_fit  = [archive.fitness;   fits];

    keys = sequences_to_keys(all_sols);
    [~, ia] = unique(keys, 'stable');
    all_sols = all_sols(ia);
    all_fit  = all_fit(ia,:);

    keep = non_dominated_filter(all_fit);
    archive.solutions = all_sols(keep);
    archive.fitness   = all_fit(keep,:);
end

function keep = non_dominated_filter(F)
    N = size(F,1);
    keep = true(N,1);
    for i = 1:N
        if ~keep(i), continue; end
        for j = 1:N
            if i==j || ~keep(j), continue; end
            if all(F(j,:) <= F(i,:)) && any(F(j,:) < F(i,:))
                keep(i)=false; break;
            end
        end
    end
end

function keys = sequences_to_keys(cells)
    N = numel(cells);
    keys = strings(N,1);
    for i = 1:N
        v = cells{i};
        if isrow(v), keys(i) = sprintf('%d_', v);
        else,        keys(i) = sprintf('%d_', v'); end
    end
end

function Fnd = extract_rank1_front(F)
    if isempty(F), Fnd = F; return; end
    keep = non_dominated_filter(F);
    Fnd = F(keep,:);
end

function hv = hypervolume_2d(Fnd, ref)
    if isempty(Fnd), hv = 0; return; end
    Fnd = sortrows(Fnd, 1);
    f2  = Fnd(:,2);
    f2m = flipud(cummin(flipud(f2)));

    hv = 0;
    prev_e = ref(2);
    for i = 1:size(Fnd,1)
        w = max(0, ref(1) - Fnd(i,1));
        h = max(0, prev_e  - f2m(i));
        hv = hv + w*h;
        prev_e = f2m(i);
    end
end


%% ============================================================================
%  PLOTS + EXPORT (unchanged)
% ============================================================================
function export_pareto_csv(archive, filename)
    if isempty(archive.fitness)
        fprintf('No Pareto points to export.\n');
        return;
    end

    F = archive.fitness;
    [~, iu] = unique(round(F,6), 'rows', 'stable');
    F = F(iu,:);
    F = sortrows(F,[1 2]);

    T = array2table(F, 'VariableNames', {'Makespan','Energy_kWh'});

    % Ensure directory exists
    [filepath, ~, ~] = fileparts(filename);
    if ~isempty(filepath) && ~exist(filepath, 'dir')
        mkdir(filepath);
    end

    try
        writetable(T, filename);
        fprintf('Exported Pareto front to %s\n', filename);
        return;
    catch ME
        warning(['writetable failed: ', ME.message]);
        
        % Try alternative location if default fails
        alt_filename = fullfile(pwd, 'pareto_front_rank1.csv');
        warning('Trying alternative location: %s', alt_filename);
        
        try
            writetable(T, alt_filename);
            fprintf('Exported Pareto front to %s\n', alt_filename);
            return;
        catch ME2
            warning(['Alternative location also failed: ', ME2.message]);
        end
    end

    % Manual fallback as last resort
    try
        % Try desktop as final fallback
        desktop_path = fullfile(getenv('USERPROFILE'), 'Desktop', 'pareto_front_rank1.csv');
        fid = fopen(desktop_path, 'w');
        
        if fid == -1
            % Try current directory
            desktop_path = fullfile(pwd, 'pareto_front_rank1.csv');
            fid = fopen(desktop_path, 'w');
        end
        
        if fid == -1
            error('Cannot open any file for writing.');
        end
        
        fprintf(fid, 'Makespan,Energy_kWh\n');
        for i = 1:size(F,1)
            fprintf(fid, '%.6f,%.6f\n', F(i,1), F(i,2));
        end
        fclose(fid);
        fprintf('Exported Pareto front to %s (manual writer)\n', desktop_path);
    catch finalErr
        warning(finalErr.identifier, 'All export attempts failed: %s', finalErr.message);
    end
end

%% ============================================================================
%  FINAL PRINT
% ============================================================================
function display_final_schedule(best_solution, problem)
    [fit, dec] = evaluate_schedule_fast(best_solution, problem);
    fprintf('\n=== FINAL SCHEDULE RESULTS ===\n');
    fprintf('Leader (best lexicographic)\n');
    fprintf('Makespan: %.0f (min)\n', fit(1));
    fprintf('Total Energy: %.3f kWh\n', fit(2));
    fprintf('Job Sequence: '); fprintf('%d ', best_solution); fprintf('\n\n');
    fprintf('Completion Times (Job x Operation):\n');
    disp(dec.ct);
end
%% ============================================================================
%  VISUALIZATION FUNCTIONS
% ============================================================================

%% 1. HYPERVOLUME PLOT
function plot_hypervolume(runtime_stats, opts)
    % PLOT_HYPERVOLUME - Plot hypervolume evolution over iterations
    %
    % Inputs:
    %   runtime_stats - Structure with hv_history field
    %   opts          - Algorithm options structure
    
    figure('Name', 'Hypervolume Evolution', 'Position', [100, 100, 800, 500]);
    
    hv = runtime_stats.hv_history;
    hv = hv(~isnan(hv));
    iterations = 1:length(hv);
    
    plot(iterations, hv, 'b-', 'LineWidth', 2);
    hold on;
    
    % Mark phase transition if available
    if isfield(opts, 'phase1_ratio') && isfield(runtime_stats, 'iteration_modes')
        phase1_end = round(opts.phase1_ratio * length(hv));
        if phase1_end < length(hv)
            plot([phase1_end, phase1_end], [min(hv), max(hv)], 'r--', 'LineWidth', 1.5);
            text(phase1_end, max(hv)*0.9, 'Phase 1→2', 'Color', 'r', 'FontSize', 10);
        end
    end
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Hypervolume', 'FontSize', 12);
    title('Hypervolume Evolution During Optimization', 'FontSize', 14);
    grid on;
    box on;
    
    % Add final value annotation
    if ~isempty(hv)
        final_hv = hv(end);
        text(length(hv)*0.7, max(hv)*0.2, sprintf('Final HV: %.4f', final_hv), ...
            'FontSize', 11, 'BackgroundColor', 'white');
    end
    
    hold off;
end

%% 2. MAKESPAN VS ENERGY (PARETO FRONT) PLOT
function plot_pareto_front(archive, runtime_stats, problem)
    % PLOT_PARETO_FRONT - Plot Pareto front (makespan vs energy)
    %
    % Inputs:
    %   archive       - Pareto archive structure
    %   runtime_stats - Runtime statistics
    %   problem       - Problem definition
    
    figure('Name', 'Pareto Front: Makespan vs Energy', 'Position', [150, 150, 900, 600]);
    
    % Extract non-dominated solutions
    if ~isempty(archive.fitness)
        F_all = archive.fitness;
        keep = non_dominated_filter(F_all);
        pareto_front = F_all(keep, :);
        pareto_front = sortrows(pareto_front, 1);
    else
        pareto_front = [];
    end
    
    pf = archive.fitness;

if isempty(pf)
    warning('Pareto front is empty — nothing to plot.');
    return;
end

hold on;
plot(pf(:,1), pf(:,2), 'ro-', ...
    'LineWidth', 2, ...
    'MarkerSize', 6, ...
    'DisplayName', 'Pareto Front');
    
    % Plot Pareto front
    if ~isempty(pareto_front)
        plot(pareto_front(:,1), pareto_front(:,2), 'r-o', ...
            'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        
        % Highlight best makespan and best energy solutions
        [best_makespan, idx_makespan] = min(pareto_front(:,1));
        [best_energy, idx_energy] = min(pareto_front(:,2));
        
        plot(best_makespan, pareto_front(idx_makespan,2), 'gs', ...
            'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 2);
        plot(pareto_front(idx_energy,1), best_energy, 'bs', ...
            'MarkerSize', 12, 'MarkerFaceColor', 'b', 'LineWidth', 2);
        
        % Add annotations
        text(best_makespan, pareto_front(idx_makespan,2)*1.02, ...
            sprintf('Best Cmax: %.0f', best_makespan), ...
            'FontSize', 10, 'Color', 'g', 'HorizontalAlignment', 'center');
        text(pareto_front(idx_energy,1)*1.01, best_energy, ...
            sprintf('Best Energy: %.2f kWh', best_energy), ...
            'FontSize', 10, 'Color', 'b', 'VerticalAlignment', 'bottom');
        
        % Calculate and display crowding distances
        if size(pareto_front, 1) > 2
            fronts = {1:size(pareto_front,1)};
            crowd = crowding_distance(pareto_front, fronts);
            
            % Annotate most diverse solutions
            [~, idx_diverse] = max(crowd);
            plot(pareto_front(idx_diverse,1), pareto_front(idx_diverse,2), 'md', ...
                'MarkerSize', 12, 'MarkerFaceColor', 'm', 'LineWidth', 2);
            text(pareto_front(idx_diverse,1), pareto_front(idx_diverse,2)*0.98, ...
                'Most Diverse', 'FontSize', 9, 'Color', 'm');
        end
    end
    
    % Plot reference lines (if target makespan exists)
    if isfield(runtime_stats, 'best_cmax_history') && ~isnan(runtime_stats.best_cmax_history(end))
        best_cmax = runtime_stats.best_cmax_history(end);
        y_lim = ylim;
        plot([best_cmax, best_cmax], [y_lim(1), y_lim(2)], 'k--', 'LineWidth', 1);
        text(best_cmax, y_lim(2)*0.95, sprintf('Best Cmax: %.0f', best_cmax), ...
            'FontSize', 10, 'Color', 'k', 'Rotation', 90);
    end
    
    xlabel('Makespan (minutes)', 'FontSize', 12);
    ylabel('Total Energy Consumption (kWh)', 'FontSize', 12);
    title(sprintf('Pareto Front - %dx%d Job Shop', problem.num_jobs, problem.num_machines), ...
        'FontSize', 14);
    legend('All Solutions', 'Pareto Front', 'Best Makespan', 'Best Energy', ...
        'Location', 'best');
    grid on;
    box on;
    
    hold off;
end

%% 3. GANTT CHART PLOT
function plot_gantt_chart(solution, problem, show_energy)
    % PLOT_GANTT_CHART - Plot Gantt chart for a schedule with energy visualization
    %
    % Inputs:
    %   solution    - Job sequence
    %   problem     - Problem definition
    %   show_energy - Boolean, if true shows energy states
    
    if nargin < 3
        show_energy = false;
    end
    
    % Decode schedule
    eval_counter.count = 0;
    eval_counter.max_evaluations = inf;   % disable stopping logic

    [fitness, dec] = evaluate_schedule_fast(solution, problem, eval_counter);
    makespan = fitness(1);
    
    % Setup figure
    if show_energy
        figure('Name', 'Gantt Chart with Energy States', 'Position', [200, 200, 1200, 700]);
        subplot(2,1,1);
    else
        figure('Name', 'Gantt Chart', 'Position', [200, 200, 1000, 600]);
    end
    
    % Colors for jobs
    colors = lines(problem.num_jobs);
    
    % Plot operations
    hold on;
    for j = 1:problem.num_jobs
        for o = 1:problem.num_machines
            m = problem.machine_sequences(j, o);
            start_time = dec.st(j, o);
            duration = problem.processing_times(j, m);
            
            if ~isnan(start_time)
                % Plot operation rectangle
                rectangle('Position', [start_time, m-0.4, duration, 0.8], ...
                    'FaceColor', colors(j,:), 'EdgeColor', 'k', 'LineWidth', 1);
                
                % Add job-operation label
                text(start_time + duration/2, m, sprintf('J%d-O%d', j, o), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'FontSize', 8, 'Color', 'white', 'FontWeight', 'bold');
            end
        end
    end
    
    % Configure axes
    xlabel('Time (minutes)', 'FontSize', 11);
    ylabel('Machine', 'FontSize', 11);
    title(sprintf('Gantt Chart - Makespan: %.0f min, Energy: %.2f kWh', ...
        makespan, fitness(2)), 'FontSize', 13);
    
    ylim([0.5, problem.num_machines + 0.5]);
    xlim([0, makespan * 1.05]);
    yticks(1:problem.num_machines);
    yticklabels(arrayfun(@(x) sprintf('M%d', x), 1:problem.num_machines, 'UniformOutput', false));
    
    grid on;
    box on;
    
    % Add legend
    h = zeros(problem.num_jobs, 1);
    for j = 1:min(problem.num_jobs, 10)  % Limit legend entries
        h(j) = plot(NaN, NaN, 's', 'MarkerFaceColor', colors(j,:), ...
            'MarkerEdgeColor', 'k', 'MarkerSize', 10);
    end
    legend(h(1:min(problem.num_jobs, 10)), arrayfun(@(x) sprintf('Job %d', x), ...
        1:min(problem.num_jobs, 10), 'UniformOutput', false), ...
        'Location', 'eastoutside');
    
    hold off;
    
    %% Energy states subplot
    if show_energy
        subplot(2,1,2);
        
        % Calculate machine states over time
        time_resolution = 5; % minutes per sample
        time_points = 0:time_resolution:makespan;
        states = zeros(problem.num_machines, length(time_points));
        
        for m = 1:problem.num_machines
            iv = dec.mach_iv{m};
            if isempty(iv)
                states(m, :) = 3; % Always sleeping
                continue;
            end
            
            for t_idx = 1:length(time_points)
                t = time_points(t_idx);
                active = false;
                
                % Check if machine is processing
                for k = 1:size(iv, 1)
                    if t >= iv(k,1) && t < iv(k,2)
                        active = true;
                        break;
                    end
                end
                
                if active
                    states(m, t_idx) = 1; % Processing
                else
                    % Check idle vs sleep
                    if t < iv(1,1) || t > iv(end,2)
                        % Before first or after last operation
                        states(m, t_idx) = 3; % Sleeping
                    else
                        % Check gaps between operations
                        in_gap = true;
                        for k = 1:size(iv,1)-1
                            if t >= iv(k,2) && t < iv(k+1,1)
                                gap = iv(k+1,1) - iv(k,2);
                                if gap >= problem.energy.sleep_threshold_min
                                    states(m, t_idx) = 3; % Sleeping
                                else
                                    states(m, t_idx) = 2; % Idle
                                end
                                in_gap = false;
                                break;
                            end
                        end
                        if in_gap
                            states(m, t_idx) = 2; % Idle
                        end
                    end
                end
            end
        end
        
        % Plot energy states as heatmap
        imagesc(time_points, 1:problem.num_machines, states);
        colormap([0.9, 0.3, 0.3;    % Processing - Red
                  1.0, 0.8, 0.2;    % Idle - Yellow
                  0.3, 0.7, 0.3]);  % Sleeping - Green
        
        colorbar('Ticks', [1, 2, 3], 'TickLabels', {'Processing', 'Idle', 'Sleeping'});
        
        xlabel('Time (minutes)', 'FontSize', 11);
        ylabel('Machine', 'FontSize', 11);
        title('Machine Energy States Over Time', 'FontSize', 13);
        
        yticks(1:problem.num_machines);
        yticklabels(arrayfun(@(x) sprintf('M%d', x), 1:problem.num_machines, 'UniformOutput', false));
        
        grid on;
        box on;
    end
end

%% 4. CONVERGENCE PLOTS (MULTIPLE METRICS)
function plot_convergence(runtime_stats, opts)
    % PLOT_CONVERGENCE - Plot multiple convergence metrics
    %
    % Inputs:
    %   runtime_stats - Runtime statistics structure
    %   opts          - Algorithm options
    
    figure('Name', 'Algorithm Convergence', 'Position', [100, 100, 1200, 800]);
    
    % Subplot 1: Makespan convergence
    subplot(2,2,1);
    iterations = 1:length(runtime_stats.best_cmax_history);
    plot(iterations, runtime_stats.best_cmax_history, 'b-', 'LineWidth', 2);
    hold on;
    plot(iterations, runtime_stats.leader_cmax_history, 'r--', 'LineWidth', 1.5);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Makespan (min)', 'FontSize', 11);
    title('Makespan Convergence', 'FontSize', 12);
    legend('Best Found', 'Current Leader', 'Location', 'best');
    grid on;
    box on;
    
    % Add target line if specified
    if isfield(opts, 'stop_when_reached') && opts.stop_when_reached > 0
        yline(opts.stop_when_reached, 'g--', 'LineWidth', 1.5, ...
            'DisplayName', 'Target Makespan');
    end
    
    % Subplot 2: Energy convergence
    subplot(2,2,2);
    plot(iterations, runtime_stats.leader_energy_history, 'm-', 'LineWidth', 2);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Energy (kWh)', 'FontSize', 11);
    title('Energy Consumption Convergence', 'FontSize', 12);
    grid on;
    box on;
    
    % Subplot 3: Archive size
    subplot(2,2,3);
    plot(iterations, runtime_stats.archive_sizes, 'g-', 'LineWidth', 2);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Archive Size', 'FontSize', 11);
    title('Pareto Archive Size Evolution', 'FontSize', 12);
    grid on;
    box on;
    
    % Subplot 4: Computation time
    subplot(2,2,4);
    if isfield(runtime_stats, 'iter_times')
        plot(1:length(runtime_stats.iter_times), runtime_stats.iter_times, 'k-', 'LineWidth', 1.5);
        hold on;
        plot(1:length(runtime_stats.iter_times), ...
            movmean(runtime_stats.iter_times, 5), 'r-', 'LineWidth', 2);
        xlabel('Iteration', 'FontSize', 11);
        ylabel('Time (seconds)', 'FontSize', 11);
        title('Computation Time per Iteration', 'FontSize', 12);
        legend('Raw', 'Moving Avg (5 iters)', 'Location', 'best');
        grid on;
        box on;
    end
    
    % Add phase transition markers
    if isfield(opts, 'phase1_ratio')
        phase1_end = round(opts.phase1_ratio * max(iterations));
        for sp = 1:4
            subplot(2,2,sp);
            hold on;
            xline(phase1_end, 'r--', 'LineWidth', 1, 'Alpha', 0.5);
            if sp == 1
                text(phase1_end, max(ylim)*0.9, 'Phase 2 Start', ...
                    'Color', 'r', 'FontSize', 8, 'HorizontalAlignment', 'right');
            end
        end
    end
end

%% 5. ENERGY BREAKDOWN CHART
function plot_energy_breakdown(solution, problem)
    % PLOT_ENERGY_BREAKDOWN - Show energy consumption breakdown by type
    %
    % Inputs:
    %   solution - Job sequence
    %   problem  - Problem definition
    
    % Evaluate and get detailed energy breakdown
    eval_counter.count = 0;
    eval_counter.max_evaluations = inf;   % disable evaluation limit

    [~, dec] = evaluate_schedule_fast(solution, problem, eval_counter);
    
    % Calculate energy components for each machine
    nm = problem.num_machines;
    runtime_energy = zeros(nm, 1);
    idle_energy = zeros(nm, 1);
    sleep_energy = zeros(nm, 1);
    wakeup_energy = zeros(nm, 1);
    
    for m = 1:nm
        iv = dec.mach_iv{m};
        makespan = dec.makespan;
        
        if isempty(iv)
            % Machine never used
            sleep_energy(m) = makespan * problem.time_unit_hours * problem.energy.sleep_power(m);
            continue;
        end
        
        % Runtime energy
        runtime_min = sum(iv(:,2) - iv(:,1));
        runtime_energy(m) = runtime_min * problem.time_unit_hours * problem.runtime_power(m);
        
        % Process gaps
        % First gap
        gap = iv(1,1) - 0;
        if gap > 0
            [e_on, e_sl, e_wu] = gap_energy_components_fast(m, gap, ...
                problem.energy.sleep_threshold_min, problem.energy.sleep_power, ...
                problem.energy.wakeup_energy, problem.energy.wakeup_time_min, ...
                problem.energy.wakeup_time_min * problem.time_unit_hours, problem, false);
            idle_energy(m) = idle_energy(m) + e_on;
            sleep_energy(m) = sleep_energy(m) + e_sl;
            wakeup_energy(m) = wakeup_energy(m) + e_wu;
        end
        
        % Between operations
        for k = 1:size(iv,1)-1
            gap = iv(k+1,1) - iv(k,2);
            if gap > 0
                [e_on, e_sl, e_wu] = gap_energy_components_fast(m, gap, ...
                    problem.energy.sleep_threshold_min, problem.energy.sleep_power, ...
                    problem.energy.wakeup_energy, problem.energy.wakeup_time_min, ...
                    problem.energy.wakeup_time_min * problem.time_unit_hours, problem, false);
                idle_energy(m) = idle_energy(m) + e_on;
                sleep_energy(m) = sleep_energy(m) + e_sl;
                wakeup_energy(m) = wakeup_energy(m) + e_wu;
            end
        end
        
        % Final gap
        gap = makespan - iv(end,2);
        if gap > 0
            [e_on, e_sl, e_wu] = gap_energy_components_fast(m, gap, ...
                problem.energy.sleep_threshold_min, problem.energy.sleep_power, ...
                problem.energy.wakeup_energy, problem.energy.wakeup_time_min, ...
                problem.energy.wakeup_time_min * problem.time_unit_hours, problem, true);
            idle_energy(m) = idle_energy(m) + e_on;
            sleep_energy(m) = sleep_energy(m) + e_sl;
            wakeup_energy(m) = wakeup_energy(m) + e_wu;
        end
    end
    
    % Plot stacked bar chart
    figure('Name', 'Energy Consumption Breakdown', 'Position', [150, 150, 1000, 600]);
    
    energy_matrix = [runtime_energy, idle_energy, sleep_energy, wakeup_energy];
    h = bar(energy_matrix, 'stacked');
    
    % Set colors
    colors = [0.9, 0.3, 0.3;    % Runtime - Red
              1.0, 0.8, 0.2;    % Idle - Yellow
              0.3, 0.7, 0.3;    % Sleep - Green
              0.3, 0.5, 0.9];   % Wakeup - Blue
    
    for i = 1:4
        h(i).FaceColor = colors(i,:);
        h(i).EdgeColor = 'k';
    end
    
    xlabel('Machine', 'FontSize', 12);
    ylabel('Energy Consumption (kWh)', 'FontSize', 12);
    title('Energy Consumption Breakdown by Machine and State', 'FontSize', 14);
    
    legend({'Runtime', 'Idle', 'Sleep', 'Wakeup'}, 'Location', 'northoutside', ...
        'Orientation', 'horizontal');
    
    grid on;
    box on;
    
    % Add total energy annotation
    total_energy = sum(sum(energy_matrix));
    text(0.5, max(ylim)*0.95, sprintf('Total Energy: %.2f kWh', total_energy), ...
        'FontSize', 12, 'BackgroundColor', 'white');
    
    % Add machine utilization percentages
    util_percent = zeros(nm, 1);
    for m = 1:nm
        if ~isempty(dec.mach_iv{m})
            total_runtime = sum(dec.mach_iv{m}(:,2) - dec.mach_iv{m}(:,1));
            util_percent(m) = (total_runtime / dec.makespan) * 100;
        end
    end
    
    % Add utilization as text above bars
    for m = 1:nm
        text(m, sum(energy_matrix(m,:))*1.02, sprintf('%.1f%%', util_percent(m)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end

%% 6. WRAPPER FUNCTION TO GENERATE ALL CHARTS
function generate_all_charts(best_solution, archive, runtime_stats, problem, opts)
    % GENERATE_ALL_CHARTS - Generate all visualization charts
    %
    % Inputs:
    %   best_solution  - Best solution found
    %   archive        - Pareto archive
    %   runtime_stats  - Runtime statistics
    %   problem        - Problem definition
    %   opts           - Algorithm options
    
    fprintf('\n=== GENERATING VISUALIZATIONS ===\n');
    
    % 1. Hypervolume evolution
    try
        plot_hypervolume(runtime_stats, opts);
        fprintf('✓ Hypervolume plot created\n');
    catch ME
        fprintf('✗ Error creating hypervolume plot: %s\n', ME.message);
    end
    
    % 2. Pareto front
    try
        plot_pareto_front(archive, runtime_stats, problem);
        fprintf('✓ Pareto front plot created\n');
    catch ME
        fprintf('✗ Error creating Pareto front plot: %s\n', ME.message);
    end
    
    % 3. Convergence plots
    try
        plot_convergence(runtime_stats, opts);
        fprintf('✓ Convergence plots created\n');
    catch ME
        fprintf('✗ Error creating convergence plots: %s\n', ME.message);
    end
    
    % 4. Gantt chart (without energy states)
    try
        plot_gantt_chart(best_solution, problem, false);
        fprintf('✓ Gantt chart created\n');
    catch ME
        fprintf('✗ Error creating Gantt chart: %s\n', ME.message);
    end
    
    % 5. Gantt chart with energy states (optional, can be slow)
    if problem.num_jobs * problem.num_machines <= 225  % Limit for performance
        try
            plot_gantt_chart(best_solution, problem, true);
            fprintf('✓ Gantt chart with energy states created\n');
        catch ME
            fprintf('✗ Error creating detailed Gantt chart: %s\n', ME.message);
        end
    end
    
    % 6. Energy breakdown
    try
        plot_energy_breakdown(best_solution, problem);
        fprintf('✓ Energy breakdown plot created\n');
    catch ME
        fprintf('✗ Error creating energy breakdown plot: %s\n', ME.message);
    end
    
    fprintf('\n=== VISUALIZATION COMPLETE ===\n');
end
