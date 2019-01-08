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
    byhand1 = load(strcat('data/byhand_1_', N, '.txt'));
    byhand2 = load(strcat('data/byhand_2_', N, '.txt'));
    byhand3 = load(strcat('data/byhand_4_', N, '.txt'));
    full1 = load(strcat('data/full_1_', N, '.txt'));
    full2 = load(strcat('data/full_2_', N, '.txt'));
    full3 = load(strcat('data/full_4_', N, '.txt'));
    sparse3 = load(strcat('data/sparse_4_', N, '.txt'));
    byhand1 = byhand1 / byhand1(1);
    byhand2 = byhand2 / byhand2(1);
    byhand3 = byhand3 / byhand3(1);  % TODO: Vectorise these expressions
    full1 = full1 / full1(1);
    full2 = full2 / full2(1);
    full3 = full3 / full3(1);
    sparse3 = sparse3 / sparse3(1);
    mat1 = [byhand1(2), byhand1(5);
            byhand2(2), byhand2(5);
            byhand3(2), byhand3(5);
            full1(2), full1(5);
            full2(2), full2(5);
            full3(2), full3(5);
            sparse3(2), sparse3(5)];
    mat2 = [byhand1(3), byhand1(4), byhand1(6), byhand1(7);
            byhand2(3), byhand2(4), byhand2(6), byhand2(7);
            byhand3(3), byhand3(4), byhand3(6), byhand3(7);
            full1(3), full1(4), full1(6), full1(7);
            full2(3), full2(4), full2(6), full2(7);
            full3(3), full3(4), full3(6), full3(7);
            sparse3(3), sparse3(4), sparse3(6), sparse3(7)];
    mat3 = [0, 0, full1(8), 0;
            0, 0, full2(8), 0;
            0, 0, full3(8), 0;
            sparse3(8), sparse3(9), sparse3(10), sparse3(11)];
    xtick = ['Hand / 1';'Hand / 2';'Hand / 4';'Full / 1';'Full / 2';'Full / 4';'Sparse / 4'];
else
    byhand1 = load(strcat('data/byhand_4_', N, '.txt'));
    byhand2 = load(strcat('data/byhand_16_', N, '.txt'));
    byhand3 = load(strcat('data/byhand_64_', N, '.txt'));
    sparse1 = load(strcat('data/sparse_4_', N, '.txt'));
    sparse2 = load(strcat('data/sparse_16_', N, '.txt'));
    sparse3 = load(strcat('data/sparse_64_', N, '.txt'));
    byhand1 = byhand1 / byhand1(1);
    byhand2 = byhand2 / byhand2(1);
    byhand3 = byhand3 / byhand3(1);  % TODO: Vectorise these expressions
    sparse1 = sparse1 / sparse1(1);
    sparse2 = sparse2 / sparse2(1);
    sparse3 = sparse3 / sparse3(1);
    mat1 = [byhand1(2), byhand1(5);
            byhand2(2), byhand2(5);
            byhand3(2), byhand3(5);
            sparse1(2), sparse1(5);
            sparse2(2), sparse2(5);
            sparse3(2), sparse3(5)];
    mat2 = [byhand1(3), byhand1(4), byhand1(6), byhand1(7);
            byhand2(3), byhand2(4), byhand2(6), byhand2(7);
            byhand3(3), byhand3(4), byhand3(6), byhand3(7);
            sparse1(3), sparse1(4), sparse1(6), sparse1(7);
            sparse2(3), sparse2(4), sparse2(6), sparse2(7);
            sparse3(3), sparse3(4), sparse3(6), sparse3(7)];
    mat3 = [sparse1(8), sparse1(9), sparse1(10), sparse1(11);
            sparse2(8), sparse2(9), sparse2(10), sparse2(11);
            sparse3(8), sparse3(9), sparse3(10), sparse3(11)]; % TODO: These should be proportion of Jacobian evaluation
    xtick = ['Hand / 4';'Hand / 16';'Hand / 64';'Sparse / 4';'Sparse / 16';'Sparse / 64'];
end


% TSStep vs. TSAdjointStep
b = bar(mat1, 'stacked');
l = legend('TSStep', 'TSAdjointStep', 'Other');
set(l, 'Position', [0.75 0.7 0.19 0.14]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of runtime');
set(gca, 'xticklabel', xtick);
pbaspect([3, 1]);
saveas(gcf, outfile = strcat('plots/steps', N, '.pdf'));

hold off

% Subcomponents
b = bar(mat2, 'stacked');
l = legend('TSFunctionEval', 'TSJacobianEval', 'TSTrajectorySet', 'TSTrajectoryGet');
set(l, 'Position', [0.75 0.7 0.19 0.14]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of runtime');
set(gca, 'xticklabel', xtick);
pbaspect([3, 1])
saveas(gcf, outfile = strcat('plots/comps', N, '.pdf'));

% ADOL-C parts
b = bar(mat3, 'stacked');
l = legend('SparsityPattern', 'Colouring', 'Propagation', 'Recovery', 'Other');
set(l, 'Position', [0.75 0.7 0.19 0.14]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of runtime');
if n == 65
    xtick = ['Full / 1';'Full / 2';'Full / 4';'Sparse / 4'];
else
    xtick = ['Sparse / 4';'Sparse / 16';'Sparse / 64'];
end
set(gca, 'xticklabel', xtick);
pbaspect([3, 1]);
saveas(gcf, outfile = strcat('plots/adolc', N, '.pdf'));

end
