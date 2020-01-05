function col=swapcolor()
a=get(0,'default');
if isequal(a.defaultAxesColor,[0 0 0]);
    col='w';
else
    col='k'
end
