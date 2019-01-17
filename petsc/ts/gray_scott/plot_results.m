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
toplot = byhand(idx,2);
graph0 = loglog(ncores(idx,1), toplot, 'k-+');
if n == 65
    text(ncores(idx,1), toplot-8./ncores(idx,1), cellstr(int2str(byhand(idx,3))), 'VerticalAlignment','cap', 'Color', 'k')
else
    text(ncores(idx,1), toplot, cellstr(int2str(byhand(idx,3))), 'VerticalAlignment','cap', 'Color', 'k')
end
set(graph0, 'LineWidth', 1.5);

hold on

if n == 65
    full = load(strcat('data/full', N, '.txt'));
    idx = ~isnan(full(:, 2));
    %toplot = full(idx,2)./byhand(idx,2)
    toplot = full(idx,2)
    graph1 = loglog(ncores(idx,1),toplot, 'b-^');
    text(ncores(idx,1), toplot, cellstr(int2str(full(idx,3))), 'VerticalAlignment','cap', 'Color', [0, 0.4470, 0.7410])
    set(graph1, 'LineWidth', 1.5);
end

hold on

sparse = load(strcat('data/sparse', N, '.txt'));
idx = ~isnan(sparse(:, 2));
%toplot = sparse(idx, 2)./byhand(idx,2)
toplot = sparse(idx, 2)
graph2 = loglog(ncores(idx, 1), toplot, 'm-o');
text(ncores(idx,1), toplot, cellstr(int2str(sparse(idx,3))), 'VerticalAlignment','cap', 'Color', 	[0.75, 0, 0.75])
set(graph2, 'LineWidth', 1.5);

matfree = load(strcat('data/matfree', N, '.txt'));
idx = ~isnan(matfree(:, 2));
%toplot = matfree(idx, 2)./byhand(idx,2)
toplot = matfree(idx, 2)
graph3 = loglog(ncores(idx, 1), toplot, 'r-*');
text(ncores(idx,1), toplot, cellstr(int2str(matfree(idx,3))), 'VerticalAlignment','cap', 'Color', [0.6350, 0.0780, 0.1840])
set(graph3, 'LineWidth', 1.5);

grid on
pbaspect([1 2])

if n == 65
    %l = legend('Dense', 'Sparse', 'Matrix-free');
    l = legend('Analytic', 'Dense', 'Sparse', 'Matrix-free');
    %set(l, 'Position', [0.465 0.8 0.2 0.11]);
    set(l, 'Position', [0.515 0.8 0.15 0.11]);
    xlim([1,8]);
    xtick = [1 2 4 8];
    xticklabel = ['1';'2';'4';'8'];
    %ylim([0.5,500]);
    ytick = [1 10 100 1000 10000];
    yticklabel = ['10^0';'10^1';'10^2';'10^3';'10^4'];
else
    %l = legend('Sparse', 'Matrix-free');
    l = legend('Analytic', 'Sparse', 'Matrix-free');
    %set(l, 'Position', [0.37 0.8 0.2 0.11]);
    set(l, 'Position', [0.515 0.8 0.15 0.11]);
    xlim([4,64]);
    xtick = [4 16 64];
    xticklabel = ['4';'16';'64'];
    %ytick = [1 2 4 8];
    %yticklabel = ['1';'2';'4';'8'];
    ytick = [100 1000 10000 100000];
    yticklabel = ['10^2';'10^3';'10^4';'10^5'];
    %ylim([1,4]);
end
%set(l, 'FontSize', 10);
set(gca, 'xtick', xtick);
set(gca, 'ytick', ytick);
set(gca, 'xticklabel', xticklabel);
set(gca, 'yticklabel', yticklabel);
xl = xlabel('Number of processors');
set(xl, 'FontSize', 14);
%yl = ylabel('Relative runtime');
yl = ylabel('Runtime (s)');
set(yl, 'FontSize', 14);
%title(strcat('Gray-Scott problem solved on a ', {' '}, N, 'x', N, ' grid'))
saveas(gcf, outfile = strcat('plots/res', N, '.pdf'));

end
