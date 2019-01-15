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
    mat2 = [dat1(:, [3, 4]), ones(size(dat1(:, 1))) - dat1(:, 3) - dat1(:, 4)];

    % Data normalised by TSJacobianEval timing
    mat3 = zeros(4, 4);
    mat3(1, :) = full1(8:11) / sum(full1(8:11));
    mat3(2, :) = full2(8:11) / sum(full2(8:11));
    mat3(3, :) = full3(8:11) / sum(full3(8:11));
    mat3(4, :) = sparse3(8:11) / sum(sparse3(8:11));
    xtick0 = ['1';'2';'4';'1';'2';'4';'4'];
    xtick1 = ['Analytic';'Analytic';'Analytic';'Dense';'Dense';'Dense';'Sparse'];
    xtick2 = ['1';'2';'4';'4'];
    xtick3 = ['Dense';'Dense';'Dense';'Sparse'];
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
    mat2 = [dat1(:, [3, 4]), ones(size(dat1(:, 1))) - dat1(:, 3) - dat1(:, 4)];

    % Data normalised by TSJacobianEval timing
    mat3 = zeros(3, 4);
    mat3(1, :) = sparse1(8:11) / sum(sparse1(8:11));
    mat3(2, :) = sparse2(8:11) / sum(sparse2(8:11));
    mat3(3, :) = sparse3(8:11) / sum(sparse3(8:11));
    xtick0 = ['4';'16';'64';'4';'16';'64'];
    xtick1 = ['Analytic';'Analytic';'Analytic';'Sparse';'Sparse';'Sparse'];
    xtick2 = ['4';'16';'64'];
    xtick3 = ['Sparse';'Sparse';'Sparse'];
end

% TSStep vs. TSAdjointStep
b = bar(mat1, 'stacked');
l = legend('TSStep', 'TSAdjointStep');
set(l, 'FontSize', 13);
set(l, 'Position', [0.716 0.7 0.19 0.11]);
xl = xlabel('Jacobian computation strategy / Number of processors');
yl = ylabel('Proportion of runtime');
set(xl, 'FontSize', 12);
set(yl, 'FontSize', 12);
set(gca, 'xticklabel', xtick0);
pbaspect([3, 1]);
saveas(gcf, outfile = strcat('plots/steps', N, '.pdf'));

hold off

% Subcomponents
b = bar(mat2, 'stacked');
set(b,{'FaceColor'},{[0.6350, 0.0780, 0.1840];[0.3010, 0.7450, 0.9330];[0.25, 0.25, 0.25]});
%l = legend('TSFunctionEval', 'TSJacobianEval', 'Other');
%set(l, 'FontSize', 12);
%set(l, 'Position', [0.85 0.85 0.2 0.13]);
xl = xlabel('Number of processors');
yl = ylabel('Proportion of runtime');
set(xl, 'FontSize', 12);
set(yl, 'FontSize', 12);
set(gca, 'xticklabel', xtick0);
hText = text(1:size(mat2), ones(size(xtick1), 1) + 0.07, xtick1, 'rotation', 90);
set(hText, 'VerticalAlignment','bottom', 'HorizontalAlignment', 'center','FontSize',12, 'Color','k');
pbaspect([1 ,2])
saveas(gcf, outfile = strcat('plots/comps', N, '.pdf'));

% ADOL-C parts
b = bar(mat3, 'stacked');
%l = legend('SparsityPattern', 'Colouring', 'Propagation', 'Recovery');
%set(l, 'FontSize', 13);
%set(l, 'Position', [0.15 0.73 0.205 0.19]);
xl = xlabel('Number of processors');
yl = ylabel('Relative runtime');
set(xl, 'FontSize', 12);
set(yl, 'FontSize', 12);
set(gca, 'xticklabel', xtick2);
hText = text(1:size(mat3), ones(size(xtick3), 1) + 0.025, xtick3);
set(hText, 'VerticalAlignment','bottom', 'HorizontalAlignment', 'center','FontSize',12, 'Color','k');
pbaspect([2, 1]);
saveas(gcf, outfile = strcat('plots/adolc', N, '.pdf'));

end
