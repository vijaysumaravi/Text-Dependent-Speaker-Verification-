x = categorical({'SVM (Train: Clean)','SVM (Train: Multi)'});
x = reordercats(x,{'SVM (Train: Clean)','SVM (Train: Multi)'});
y = [
    9.9 7.9 6.6 9.9;
    21  13.8    13.2    22.8;
    ];
bar(x,y)
colormap(lines(4))
title('Training (FPR+FNR)/2')
legend('location','eastoutside','Core','Core+GFCC','Core+GFCC+LPC','Core+LPC')
% ylim([0 30])

% y = [20.9   13.7    13.1    12.8;
%     21.0    13.8    13.2    12.9];
% plot([1:4],y,'-*','LineWidth',3)
% xticks([1 2 3 4])
% xticklabels({'Core','+GFCC','+LPC','+PLP'})
% axis([0 5 12 22])
% title('SVM Training Performance')
% legend('location','eastoutside','FPR','FNR')
