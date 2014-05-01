(** Dennis Lu, Julie Chang **)
(** CS51 Final Project Spring 2014 **)

open Core.Std
open Types
open Heuristics
open DTrees

exception NODATA

(* Opens the file that results will be written into *)
let oc = open_out "results.txt"
let _  = fprintf oc "%s\n\n" "Dennis Lu, Julie Chang - CS51 Final Project"


(********** Example 1 : Will you play tennis today? **********)
let _  = fprintf oc "%s\n\n" "Part 1: Will you play tennis today?"

let attr_and_vals = [("outlook", ["sunny"; "overcast"; "rain"]);
		     ("temperature", ["hot"; "mild"; "cool"]);
		     ("humidity", ["normal"; "high"]);
		     ("wind", ["weak"; "strong"])]

(* Sample datapoints for the tennis playing decision tree *)
(* Borrowed from http://www.cs.princeton.edu/courses/archive/spr07/
   cos424/papers/mitchell-dectrees.pdf *)
let d1 = {attributes = [("outlook", "sunny"); ("temperature", "hot"); 
    ("humidity", "high"); ("wind", "weak")]; classification = false}
let d2 = {attributes = [("outlook", "sunny"); ("temperature", "hot"); 
    ("humidity", "high"); ("wind", "strong")]; classification = false}
let d3 = {attributes = [("outlook", "overcast"); ("temperature", "hot"); 
    ("humidity", "high"); ("wind", "weak")]; classification = true}
let d4 = {attributes = [("outlook", "rain"); ("temperature", "mild"); 
    ("humidity", "high"); ("wind", "weak")]; classification = true}
let d5 = {attributes = [("outlook", "rain"); ("temperature", "cool"); 
    ("humidity", "normal"); ("wind", "weak")]; classification = true}
let d6 = {attributes = [("outlook", "rain"); ("temperature", "cool"); 
    ("humidity", "normal"); ("wind", "strong")]; classification = false}
let d7 = {attributes = [("outlook", "overcast"); ("temperature", "cool"); 
    ("humidity", "normal"); ("wind", "strong")]; classification = true}
let d8 = {attributes = [("outlook", "sunny"); ("temperature", "mild"); 
    ("humidity", "high"); ("wind", "weak")]; classification = false}
let d9 = {attributes = [("outlook", "sunny"); ("temperature", "cool"); 
    ("humidity", "normal"); ("wind", "weak")]; classification = true}
let d10 = {attributes = [("outlook", "rain"); ("temperature", "mild"); 
    ("humidity", "normal"); ("wind", "weak")]; classification = true}
let d11 = {attributes = [("outlook", "sunny"); ("temperature", "mild"); 
    ("humidity", "normal"); ("wind", "strong")]; classification = true}
let d12 = {attributes = [("outlook", "overcast"); ("temperature", "mild"); 
    ("humidity", "high"); ("wind", "strong")]; classification = true}
let d13 = {attributes = [("outlook", "overcast"); ("temperature", "hot"); 
    ("humidity", "normal"); ("wind", "weak")]; classification = true}
let d14 = {attributes = [("outlook", "rain"); ("temperature", "mild"); 
    ("humidity", "high"); ("wind", "strong")]; classification = false}
    
let dataset = [d1; d2; d3; d4; d5; d6; d7; d8; d9; d10; d11; d12; d13; d14]

(* Test data for predictions *)
let t1 = [("outlook", "sunny"); ("temperature", "hot"); 
    ("humidity", "normal"); ("wind", "strong")]
let t2 = [("outlook", "sunny"); ("temperature", "hot"); 
    ("humidity", "high"); ("wind", "weak")]
let t3 = [("outlook", "overcast"); ("temperature", "hot"); 
    ("humidity", "high"); ("wind", "weak")]
let t4 = [("outlook", "rain"); ("temperature", "mild"); 
    ("humidity", "high"); ("wind", "strong")]

(* Builds tennis playing tree with ID3 algorithm *)
module InfoGainID3Tree = ID3Tree(InfoGain)
let built_id3 = InfoGainID3Tree.build dataset attr_and_vals

let _  = fprintf oc "%s\n" "ID3 Tree:"
let _ = InfoGainID3Tree.print_tree built_id3 oc

(* Prunes tennis playing tree with rule post-pruning *)
module Pruner = C45Tree (InfoGainID3Tree)
let rules = Pruner.build dataset attr_and_vals

let _  = fprintf oc "%s\n" "Pruned Rules:"
let _ = Pruner.print_rules rules oc

(*****************************************************************************)
(* Functions to read and work with data from a file *)

(* Reads a text file into a list of strings *)
let read_file (filename: string) : string list =
  let lines = ref [] in
  let chan = open_in filename in
  try 
    while true; do
      lines := input_line chan :: !lines
    done; []
  with End_of_file ->
    close_in_noerr chan;
    List.rev !lines

(* Converts a properly formatted string into a datapoint.
 * features is a list of attributes.
 * s should take the form val1,val2,val3,... with no extra spaces. *)
let str_to_datapoint (features: string list) (s: string): datapoint = 
  let values_and_eval = String.split s ~on:',' in
  let reversed = List.rev values_and_eval in
  let last = match List.hd reversed with
    | None -> raise NODATA
    | Some x -> x
  in
  let values = match List.tl reversed with
    | None -> raise NODATA
    | Some x -> List.rev x
  in
  let eval = not (last = "unacc") in
  match List.zip features values with
    | None -> failwith "Improper data"
    | Some x -> {attributes = x; classification = eval}

(* Splits a complete dataset into training and testing groups. 
 * p is the probability a datapoint will be put into the training group. *)
let split_dataset (data: datapoint list) (p: float) : 
    (datapoint list) * (datapoint list) =
  List.fold_left ~init:([], []) ~f:(fun (x, y) b -> 
    if Random.float 1. <= p then (b::x, y) else (x, b::y)) data

(* Calculates the prediction accuracy of a tree and the pruned tree given
 * a set of testing data *)
let get_accuracy (data: datapoint list) (t: tree) (r: rule list) :
  float * float =
  let id3_correct = List.fold_left ~init:0  ~f:(fun a b -> 
    if InfoGainID3Tree.predict t b.attributes = b.classification then a + 1
    else a) data in 
  let pruned_correct = List.fold_left ~init:0  ~f:(fun a b -> 
    if Pruner.predict r b.attributes = b.classification then a + 1
    else a) data in
  let n = Float.of_int (List.length data) in
  (Float.of_int id3_correct) /. n, (Float.of_int pruned_correct) /. n

(********** Example: Car Evaluation **********)
(* Attributes   Values
   -----------------------------------
   buying       vhigh, high, med, low
   maint        vhigh, high, med, low
   doors        2, 3, 4, 5-more
   persons      2, 4, more
   lug_boot     small, med, big
   safety       low, med, high 

 * A car can be evaluated as unacc, acc, good, or vgood
 * We consider unacc to be a failed eval (outcome = true) 
 * and the others to be a passed eval (outcome = false) *)

let _  = fprintf oc "\n%s\n\n" "Part 2: Car Evaluation"

let car_a_v = [("buying", ["vhigh"; "high"; "med"; "low"]);
		("maint", ["vhigh"; "high"; "med"; "low"]);
		("doors", ["2"; "3"; "4"; "5more"]);
		("persons", ["2"; "4"; "more"]);
		("lug_boot", ["small"; "med"; "big"]);
		("safety", ["low"; "med"; "high"])]
		 
(* Reads in car data *)
let car_data = read_file "car.txt"

let features = ["buying"; "maint"; "doors"; "persons"; "lug_boot"; "safety"]

let car_dataset = List.map ~f:(str_to_datapoint features) car_data

let training, testing = split_dataset car_dataset 0.2

(* Builds car evaluation decision tree *) 
let car_id3 = InfoGainID3Tree.build training car_a_v

let car_pre_prune = Pruner.tree_to_rules car_id3

(* Prunes car evaluation decision tree *)
let car_pruned = Pruner.build training car_a_v

let _  = fprintf oc "%s\n" "ID3 Tree:"
let _ = InfoGainID3Tree.print_tree car_id3 oc

let _  = fprintf oc "%s\n" "Pre-Pruned Rules (for comparison):"
let _ = Pruner.print_rules car_pre_prune oc

let _  = fprintf oc "\n%s\n" "Pruned Rules:"
let _ = Pruner.print_rules car_pruned oc
 
(* Calculates prediction accuracies for comparison *)
let id3_acc, prune_acc = get_accuracy testing car_id3 car_pruned

let _ = fprintf oc "\n%s\n" ("ID3 accuracy: " ^ (Float.to_string (id3_acc *. 100.)) ^
  "%\nPruning accuracy: " ^ (Float.to_string (prune_acc *. 100.)) ^ "%\n")

let _ = close_out_noerr oc
