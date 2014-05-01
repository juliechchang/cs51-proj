(** Dennis Lu, Julie Chang **)
(** CS51 Final Project **)
(** Heuristics.ml contains the module for calculating the heuristic 
    used to choose the next attribute of a decision tree **)

open Core.Std
open Types

module type HEURISTIC = 
sig
    val compute : datapoint list -> (datapoint list * value) list -> float
end

module InfoGain : HEURISTIC = 
struct
    (* Returns proportion of outcomes that have a positive outcome *)
    let pos_prop (data: datapoint list) : float = 
        let count = List.fold_left ~init:0 
                    ~f:(fun a b -> if b.classification then a + 1 else a) 
                    data in
        let data_size = Float.of_int (List.length data) in
        if data_size = 0. then 0.
        else (Float.of_int count) /. data_size

    (* Computes entropy of a dataset *)
    let entropy (data: datapoint list) : float = 
        let p = pos_prop data in
        let q = 1. -. p in
        if p = 0. || q = 0. then 0.
        else
            ~-. (p *. log p /. log 2. +. q *. log q /. log 2.)

    (* Computes the information gain of a set of data for a given attribute *)
    let compute (data: datapoint list) (parts: (datapoint list * value) list) :
      float = 
        let s = entropy data in
        let n = Float.of_int (List.length data) in
        let reduction = List.fold_left ~init:0. ~f:(fun a (lst, _) -> 
                            a +. (Float.of_int (List.length lst)) /. n
                            *. entropy lst) parts in
        s -. reduction
end
