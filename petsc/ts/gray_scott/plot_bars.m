function []=plot_bars(n)
% Plotting script for Gray-Scott problem performance breakdown
%
% To run, activate Octave and call plot_bars(n), where n is the number of grid points in
% each direction.
%
% PETSc command line options used:
% -da_grid_x n -da_grid_y n
% -pc_type none
% -ts_max_steps 100 -ts_trajectory_type memory
% -malloc_hbw
% -implicitform

clf;
set(gca, 'FontName', 'Times')
set(gca, 'FontSize', 14)
N = int2str(n);

if n == 65
    % Load data
    byhand1 = load(strcat('data/byhand_1_', N, '.txt'));
    byhand2 = load(strcat('data/byhand_2_', N, '.txt'));
    byhand3 = load(strcat('data/byhand_4_', N, '.txt'));
    full1 = load(strcat('data/full_1_', N, '.txt'));
    full2 = load(strcat('data/full_2_', N, '.txt'));
    full3 = load(strcat('data/full_4_', N, '.txt'));
    sparse3 = load(strcat('data/sparse_4_', N, '.txt'));

    % Data normalised by total runtime
    dat1 = zeros(7, 11);
    dat1(1, :) = byhand1 / byhand1(1);
    dat1(2, :) = byhand2 / byhand2(1);
    dat1(3, :) = byhand3 / byhand3(1);
    dat1(4, :) = full1 / full1(1);
    dat1(5, :) = full2 / full2(1);
    dat1(6, :) = full3 / full3(1);
    dat1(7, :) = sparse3 / sparse3(1);
    mat1 = dat1(:, [2, 5]);
    mat2 = dat1(:, [3, 4, 6, 7]);

    % Data normalised by TSJacobianEval timing
    mat3 = zeros(4, 4);
    mat3(1, :) = full1(8:11) / full1(4);
    mat3(2, :) = full2(8:11) / full2(4);
    mat3(3, :) = full3(8:11) / full3(4);
    mat3(4, :) = sparse3(8:11) / sparse3(4);
    xtick1 = ['Hand / 1';'Hand / 2';'Hand / 4';'Full / 1';'Full / 2';'Full / 4';'Sparse / 4'];
    xtick2 = ['Full / 1';'Full / 2';'Full / 4';'Sparse / 4'];
else
    % Load data
    byhand1 = load(strcat('data/byhand_4_', N, '.txt'));
    byhand2 = load(strcat('data/byhand_16_', N, '.txt'));
    byhand3 = load(strcat('data/byhand_64_', N, '.txt'));
    sparse1 = load(strcat('data/sparse_4_', N, '.txt'));
    sparse2 = load(strcat('data/sparse_16_', N, '.txt'));
    sparse3 = load(strcat('data/sparse_64_', N, '.txt'));

    % Data normalised by total runtime
    dat1 = zeros(6, 11);
    dat1(1, :) = byhand1 / byhand1(1);
    dat1(2, :) = byhand2 / byhand2(1);
    dat1(3, :) = byhand3 / byhand3(1);
    dat1(4, :) = sparse1 / sparse1(1);
    dat1(5, :) = sparse2 / sparse2(1);
    dat1(6, :) = sparse3 / sparse3(1);
    mat1 = dat1(:, [2, 5]);
    mat2 = dat1(:, [3, 4, 6, 7]);

    % Data normalised by TSJacobianEval timing
    mat3 = zeros(3, 4);
    mat3(1, :) = sparse1(8:11) / sparse1(4);
    mat3(2, :) = sparse2(8:11) / sparse2(4);
    mat3(3, :) = sparse3(8:11) / sparse3(4);
    xtick1 = ['Hand / 4';'Hand / 16';'Hand / 64';'Sparse / 4';'Sparse / 16';'Sparse / 64'];
    xtick2 = ['Sparse / 4';'Sparse / 16';'Sparse / 64'];
end

% TSStep vs. TSAdjointStep
b = bar(mat1, 'stacked');
l = legend('TSStep', 'TSAdjointStep', 'Other');
set(l, 'Position', [0.75 0.7 0.19 0.14]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of runtime');
set(gca, 'xticklabel', xtick1);
pbaspect([3, 1]);
saveas(gcf, outfile = strcat('plots/steps', N, '.pdf'));

hold off

% Subcomponents
b = bar(mat2, 'stacked');
l = legend('TSFunctionEval', 'TSJacobianEval', 'TSTrajectorySet', 'TSTrajectoryGet');
set(l, 'Position', [0.75 0.7 0.19 0.14]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of runtime');
set(gca, 'xticklabel', xtick1);
pbaspect([3, 1])
saveas(gcf, outfile = strcat('plots/comps', N, '.pdf'));

% ADOL-C parts
b = bar(mat3, 'stacked');
l = legend('SparsityPattern', 'Colouring', 'Propagation', 'Recovery', 'Other');
set(l, 'Position', [0.75 0.7 0.19 0.14]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of TSJacobianEval');
set(gca, 'xticklabel', xtick2);
pbaspect([3, 1]);
saveas(gcf, outfile = strcat('plots/adolc', N, '.pdf'));

end
