function []=plot_results(n)
% Plotting script for Gray-Scott problem times to solution
%
% To run, activate Octave and call plot_results(n), where n is the number of grid points in
% each direction.
%
% PETSc command line options used:
% -da_grid_x n -da_grid_y n
% -pc_type none
% -ts_max_steps 100 -ts_trajectory_type memory
% -malloc_hbw
% -implicitform

clf
set(gca, 'FontName', 'Times')
set(gca, 'FontSize', 14)
N = int2str(n);

byhand = load(strcat('data/byhand', N, '.txt'));
ncores = byhand(:, 1);
idx = ~isnan(byhand(:, 2));
graph0 = loglog(ncores(idx,1), byhand(idx,2), 'k-+');
set(graph0, 'LineWidth', 1.5);
hold on

if n == 65
    full = load(strcat('data/full', N, '.txt'));
    idx = ~isnan(full(:, 2));
    graph1 = loglog(ncores(idx,1),full(idx,2), 'b-^');
    set(graph1, 'LineWidth', 1.5);
end

sparse = load(strcat('data/sparse', N, '.txt'));
idx = ~isnan(sparse(:, 2));
graph2 = loglog(ncores(idx, 1), sparse(idx, 2), 'm-o');
set(graph2, 'LineWidth', 1.5);

matfree = load(strcat('data/matfree', N, '.txt'));
idx = ~isnan(matfree(:, 2));
graph3 = loglog(ncores(idx, 1), matfree(idx, 2), 'r-*');
set(graph3, 'LineWidth', 1.5);

grid on
pbaspect([4 1])

if n == 65
    l = legend('Hand-coded', 'ADOL-C', 'Sparse ADOL-C', 'Matrix-free ADOL-C');
    set(l, 'FontSize', 8);
    set(l, 'Position', [0.71 0.5 0.19 0.14]);
    xtick = [1 2 4];
    set(gca, 'xtick', xtick);
    xticklabel = ['1';'2';'4'];
    set(gca, 'xticklabel', xticklabel);
    ytick = [1 10 100 1000 10000];
    set(gca, 'ytick', ytick);
    yticklabel = ['10^0';'10^1';'10^2';'10^3';'10^4'];
    set(gca, 'yticklabel', yticklabel);
else
    l = legend('Hand-coded', 'Sparse ADOL-C', 'Matrix-free ADOL-C');
    set(l, 'FontSize', 8);
    set(l, 'Position', [0.135 0.5 0.19 0.14]);
    xtick = [4 16 64];
    set(gca, 'xtick', xtick);
    xticklabel = ['4';'16';'64'];
    set(gca, 'xticklabel', xticklabel);
    ytick = [100 1000 10000 100000];
    set(gca, 'ytick', ytick);
    yticklabel = ['10^2';'10^3';'10^4';'10^5'];
    set(gca, 'yticklabel', yticklabel);
end
xl = xlabel('Number of cores');
yl = ylabel('Runtime (s)');
%title(strcat('Gray-Scott problem solved on a ', {' '}, N, 'x', N, ' grid'))
saveas(gcf, outfile = strcat('plots/res', N, '.pdf'));

end
