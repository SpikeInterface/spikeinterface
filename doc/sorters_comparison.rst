Sorters comparison
==================

spiekinterface have comparison module that can be used for 3
distinct use cases:
  * compare the output of 2 sorters
  * compare the output of multiple sorters
  * compare to ground a ground truth dataset

Even if the global idea if this 3 cases have common method (compare spiketrain!)/
Internal implemention are sligtly diffrent depending the use.



Comparison with grund truth
---------------------------

A ground truth dataset can be a pair recording in vitro/vivo with a patch electrode
or in silico with a simulated dataset using engine like mearec.

The comparison is usefull to make some benchmark on a sorter engine.
The spikeforest portal do daily built benchmark on several sorters diffrents dataset.

To compute performences with ground truth dataset there basically 2 approach:

1 Global match then count perf
..............................


Algo:

  1. For each GT unit, make a score=num_match / num_gt with all tested units
  2. make a matrix shape (num_gt, num_tested)
  3. apply hungarin algo
  4. get the "optimal" pairing
  5. For each GT unit, for the optimal pair : do labelling all spike of GT and all spike of tested unit
  6. for each GT units count how many TP/FP/FN 

In the implementation, spike labeled TP somewhere can not be labled elsewhere.  

Pros:
  * each spike is counted only once
  * hit score near the chance level are set to zero
  * good FP estimation
  
Cons:
  * do not catch a cell is splitted in several part
    only the best math will be listed
  * more complicated implementation



2 Match indepedant then count perf
..................................


Algo:
  1. For each GT units, make a score=num_match / num_gt with all tested units
  2. For each GT units keep the best match with tested units
  3. Count TP/FP/FN for this best match pair


Pros:
  * very easy implementation
  * a hit score a GT units is totally
    independant from other units

Cons:
  * a tested unit can be matched several time
  * so spike can counted several times
  * keep many small hit score near the chance level
  * can have biased FP score
  * less robust with units a high firing rates


Actual implementation in spikecomparison
........................................

The actual implementation is mixed between the two methods and have buggs.
  

What should be improved for both
................................

  * score=num_match / num_g should depend on the chance level
  * provide several matrix confusion : score_agreement, score_patch, tp/fp/fn count...

  
  
  


