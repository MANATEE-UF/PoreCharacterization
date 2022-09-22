In1 = {'Hi','Hello'};
In2 = {'Bye','GoodBye'};

nIter = numel(In1)
    if nIter == numel(In2)
        Out = cell(nIter,2)
        for n = 1:numel(In1)
        [Out{nIter,1}, Out{nIter,1}] = strdist(In1{nIter},In2{nIter},2,1)
        end
    end