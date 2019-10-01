Sorters comparison
==================

spiekinterface have a comparison module that can be used for 3
distinct use cases:
  * compare to ground a ground truth dataset
  * compare the output of 2 sorters (symetric comparison)
  * compare the output of multiple sorters
  
  
Even if the global idea if this 3 cases have common method (compare spiketrain!)
Internal implementations are slightly different depending the use.



Comparison with ground truth
----------------------------




A ground truth dataset can be a pair recording **in vitro**, **in vivo** with a patch electrode
or **in silico** with a simulated dataset using engine like mearec.

The comparison to ground truth is useful to make benchmarks on a sorter engine.
The spikeforest portal do daily built benchmark on several sorters diffrents dataset.

See also `spikeforest notes <https://spikeforest.flatironinstitute.org/metrics>`_


  
To compute performances the work flow is the following:

Given:
  * **i = 1, ..., n_gt** the list ground truth units
  * **k = 1, ...., n_tested** the list of tested units from a sorter
  * **event_counts_GT[i]** the number of spike for each units of GT
  * **event_counts_ST[k]** the number of spike for each units of tested sorter

  1. **Matching firing events**
   
    For all pairs of GT unit and tested unit : count how many
    events coincide with a *delta_time* tolerence (0.4 ms by defualt).
    
    In short we count
    
      
    This give a matrix **match_event_count** of size *(n_gt X n_tested)* like this one.
      
    .. image:: images/spikecomparison_match_count.png
        :scale: 100 %
    
    Note that this matrix represent the number of **True positive (TP)**
    of each pairs. We can also deduce **False negative (FN)** and **False postive (FP)**
    
      * **num_tp**[i, k] = match_event_count[i, k]
      * **num_fn**[i, k] = event_counts_GT[i] - match_event_count[i, k]
      * **num_fp**[i, k] = event_counts_ST[k] - match_event_count[i, k]

   2. **Compute agreement score** 
   
    Given the **match_event_count** we compute the **agreement_score**
    matrix which a normalisation in the range [0, 1].
    This is done with:
    
      * agreement_score[i, k] = match_event_count[i, k] / (event_counts_GT[i] + event_counts_ST[k] - match_event_count[i, k])
    
    which is equivalent to:
    
      * agreement_score[i, k] = match_event_count[i, k] / (num_tp[i, k] + num_fp[i, k] + num_fn[i,k])
    
    or more pratically
    
      * agreement_score[i, k] = insertion(I, K) / union(I, K)
    
    which is also equivalent to the **accuracy**

    
    Here an example of the agreement matrix
    
    .. image:: images/spikecomparison_agreement_unordered.png
        :scale: 100 %
    
    This matrix can be ordered for better reading.
    
    .. image:: images/spikecomparison_agreement.png
        :scale: 100 %

    

   3. **Match units**
   
      During this step, given the **agreement_score** matrix each GT units is associated
      or not to a tested units.
      For matching, a **min_accuracy** threhold is needed (0.5 by default).
      Under this threshold not match is done. 
      There are 2 methods : **hugarian** match or **best** match.
      Pros and cons are discuss below.
      
      The `hugarian method <https://en.wikipedia.org/wiki/Hungarian_algorithm>`_
      optimize the best association between GT and tested units. With this method a GT unit can
      be associated only once or not and a tested units can be associated once or not.
      
      In the **best** method, each GT unit is associated to a tested unit that have
      the **best** agreement_score independently of all others units. In this method
      several tested units can be associated to a GT unit.
      
      Here an example of association with **hungarian** method, the first column is the GT unit id
      and the second column the tested unit id. -1 means no match:
      
      .. code-block::
      
          GT    TESTED
          0     49
          1     -1
          2     26
          3     44
          4     -1
          5     35
          6     -1
          7     -1
          8     42
          ...
      
      The spikeforest portal that make daily benchmark on sorter use the **best** match method.
       
   
   4. **Compute performances**
   
      With the list of matched units some performance metrics are computed.
      Given : **tp** the number of true positive events, **fp** number of false
      positive event, **fn** the number of false negative event, **num_gt** the number 
      of event of the matched tested units.
      
      For each GT units we have:
        * accuracy = tp / (tp + fn + fp)
        * recall = tp / (tp + fn)
        * precision = tp / (tp + fp)
        * false_discovery_rate = fp / (tp + fp)
        * miss_rate = fn / num_gt
      
      Theses performances can be visualised with the **confusion matrix**, where
      the last columns count **FN** and the last row count **FP**
      
    .. image:: images/spikecomparison_confusion.png
        :scale: 100 %

    
    
    Information about **hugarian** or **best** match method.
    
    
    * **hugarian**:
      
      Optimize best paring. If the matrix is square then all
      units are associated. If the matrix is rectangle, then either each line
      or each row is associated.
      A GT unit is associated maximum once.
      
      * Pros
      
        * each spike is counted only once
        * hit score near the chance level are set to zero
        * good FP estimation
      
      
      * Cons
      
        * do not catch a cell is splitted in several part only the best math will be listed
        * more complicated implementation
    
    * **best**
    
        Each GT units is associated to the tested unit that share the best **agreement score**.
        

      * Pros:
      
        * Each GT unit is matched totally independently from others units.
        * Accuracy score a GT units is totally independent from other units
        * Can enhance the "over merge pathology" of a sorter.

      * Cons:

        * a tested unit can be matched linked time
        * so some spike can counted several times
        * so can have biased FP score for units associated several times.
        * less robust with units having high firing rates

  
Compare the output of 2 sorters (symetric comparison)
-----------------------------------------------------

The comparison of two sorter is a quite similar to the procedure 
of **compare to ground truth** except that no assumption is done on
which is the ground truth.
So the procedure is:

  * **Matching firing events** : same a ground truth comparison
  * **Compute agreement score** : same a ground truth comparison
  * **Match units** : force with **hugarian** method

Of course no performances are computed but agreement matrix can be visualised.



Compare the output of multiple sorters
--------------------------------------

Comparison of multiple sorters is the following procedure:

  1. do pairwise symetric comparison
  2. construct a graph of all possible agreement across sorters and units
  3. extract agreement from graph
  4. make agreement spiketrains

  
  


