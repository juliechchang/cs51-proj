(** Dennis Lu, Julie Chang **)
(** CS51 Final Project **)
(** original.ml is the original file used to draft the interfaces and 
    core functionality of decision tree learning **)

open Core.Std

exception NODATA

type attribute = string
type value = string
type outcome = bool

type attr_list = (attribute * value) list
type datapoint = {attributes: attr_list; classification: outcome}
type rule = {conditions: attr_list; result: outcome}

(** Example: will you play tennis today? **)

let d1 = {attributes = [("outlook", "sunny"); ("temperature", "hot"); 
    ("humidity", "high"); ("wind", "weak")]; classification = false}
let d2 = {attributes = [("outlook", "sunny"); ("temperat
ure", "hot"); 
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

(** Tree **)

type tree = Leaf of bool | Branches of attribute * ((value * tree) list)

(***************************************************************************)

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
        (Float.of_int count) /. (Float.of_int (List.length data))

(*
    (* Returns proportion of outcomes that have a particular attribute-value
     * pair *)
    let prop (data: datapoint list) (av: attribute * value) : float =
        let count = List.fold_left ~init:0 
                  ~f:(fun a b -> if List.mem b.attributes av then a + 1
                                              else a) data in
        (Float.of_int count) /. (Float.of_int (List.length data))
*)
    
    (* Compute entropy of a dataset *)
    let entropy (data: datapoint list) : float = 
        let p = pos_prop data in
        let q = 1. -. p in
        if p = 0. then 
            ~-. (q *. log q /. log 2.)
        else if q = 0. then
            ~-. (p *. log p /. log 2.)
        else
            ~-. (p *. log p /. log 2. +. q *. log q /. log 2.)

    let compute (data: datapoint list) (parts: (datapoint list * value) list) :
      float = 
        let s = entropy data in
        let n = Float.of_int (List.length data) in
        let reduction = List.fold_left ~init:0. ~f:(fun a (lst, _) -> 
                            a +. (Float.of_int (List.length lst)) /. n
                            *. entropy lst) parts in
        s -. reduction
end

(***************************************************************************)

module type DTREE =
sig
(*
    type attribute
    type value
    type outcome
    type attr_list
    type datapoint
    type tree
*)
    val build : datapoint list -> tree
    val predict : tree -> attr_list -> outcome
    val print_tree : tree -> unit
end

module ID3Tree (H: HEURISTIC) : DTREE =
struct
(*
    type attribute = string
    type value = string
    type outcome = bool
    type attr_list = (attribute * value) list
    type datapoint = {attributes: attr_list; classification: outcome}
    type tree = Leaf of bool | Branches of attribute * ((value * tree) list)
*)
    (* Test whether or not dataset has same outcomes *)
    let rec same_outcomes (data: datapoint list) : bool =
        match data with
        | [] | [_] -> true
        | hd1::hd2::tl -> hd1.classification = hd2.classification 
                          && same_outcomes (hd2::tl)
        
    (* Find most common outcome *)
    let most_common (data: datapoint list) : bool =
        let rec helper (data: datapoint list) (count: int) : int =
            match data with
            | [] -> count
            | hd::tl -> if hd.classification then helper tl (count + 1) 
                        else helper tl (count - 1) 
        in (helper data 0) >= 0
    
    (* Gets list of attributes from dataset. Assumes no missing attributes *)
    let get_attributes (data: datapoint list) : attribute list = 
        match data with
        | [] -> raise NODATA
        | hd::_ -> List.map ~f:fst hd.attributes
        
    (* Check whether there are no attributes left in dataset. Assuming 
     * no missing attributes, so we just have to check first datapoint *)
    let no_attributes (data: datapoint list) : bool =
        get_attributes data = []

    (* Get value associated with attribute for a datapoint *)
    let get_value (p: datapoint) (attr: attribute) : value =
        match List.find ~f:(fun a -> fst a = attr) p.attributes with
        | None -> failwith "Attribute not found"
        | Some x -> snd x

    (* Partitions the dataset into groups split by the different
     * attribute values. Removes the attribute-value pair from the datset *)
    let partition (data: datapoint list) (attr: attribute) : 
      (datapoint list * value) list = 
        (* Add datapoint d with value v into current partition list l *)
        let rec add_data (l: (datapoint list * value) list) (d: datapoint) 
          (v: value) : (datapoint list * value) list =
            match l with
            | [] -> [([d], v)]
            | hd::tl -> if snd hd = v then (d::(fst hd), v)::tl
                        else hd::(add_data tl d v)
        in
        (* Remove attribute-value pair corresponding to attr from datapoint *)
        let remove_attr (d: datapoint) (attr: attribute) : datapoint = 
            let new_attrs = List.filter ~f:(fun x -> not (fst x = attr)) 
                            d.attributes in
            {attributes = new_attrs; classification = d.classification}
        in
        let rec helper (data: datapoint list) 
          (l: (datapoint list * value) list) : (datapoint list * value) list =
            match data with
            | [] -> l
            | hd::tl -> let v = get_value hd attr in
                        helper tl (add_data l (remove_attr hd attr) v)
        in helper data []
    
    (* Build decision tree given a dataset. First check if all outcomes
     * are the same: if they are, then we just use that outcome; if they
     * aren't, then we check if there are any attributes left. If there
     * aren't, then we use the most common outcome. Otherwise, we
     * compute the heuristic for every attribute, using the largest
     * heuristic to partition the dataset by the corresponding value.
     * We remove the attribute-value pair from each data point and 
     * recurisvely build the rest of the tree. *)
    let rec build (data: datapoint list) : tree = 
        if same_outcomes data then
            match data with
            | [] -> raise NODATA
            | hd::_ -> Leaf(hd.classification)
        else if no_attributes data then Leaf(most_common data)
        else
            let heuristic_list = List.map ~f:(fun x -> 
                                    (x, H.compute data (partition data x)))
                                (get_attributes data) in
            let next_attribute = fst (List.fold_left ~init:("", -1.) 
                                 ~f:(fun a b -> if snd b >= snd a then b 
                                                else a) 
                                 heuristic_list) in
            let parts = partition data next_attribute in
            Branches(next_attribute, List.map ~f:(fun (a, b) -> (b, build a))
                parts)
    
    (* Make classification prediction given a list of attribute-value
     * pairs *)
    let rec predict (t: tree) (attributes: attr_list) : outcome = 
        match t with
        | Leaf l -> l
        | Branches(a, lst) -> 
            let v = match List.find ~f:(fun x -> fst x = a) attributes with
                    | None -> failwith "Attribute not found"
                    | Some p -> snd p
            in
            let next_tree = match List.find ~f:(fun x -> fst x = v) lst with
                            | None -> failwith "Value not found"
                            | Some p -> snd p
            in
            predict next_tree attributes

    let print_tree (t: tree) : unit = ()
end

(*
module C45Tree (H: HEURISTIC) : DTREE = 
struct
    type attribute = string
    type value = Category of string | Number of int
    type outcome = bool
    type attr_list = (attribute * value) list
    type datapoint = {attributes: attr_list; classification: outcome}
    type tree = Leaf of bool | Branches of attribute * ((value * tree) list)
    let build (data: datapoint list) : tree = raise INCOMPLETE
    let predict (t: tree) (attributes: attr_list) : outcome = raise INCOMPLETE
    let print_tree (t: tree) : unit = ()
    let prune (t: tree) (data: datapoint list) : tree = raise INCOMPLETE
end
*)


(**************************************************************************

module C45Tree (T: DTREE) = 
struct
(*
    type attribute = string
    type value = Category of string | Number of int
    type outcome = bool
    type attr_list = (attribute * value) list
    type datapoint = {attributes: attr_list; classification: outcome}
    type tree = Leaf of bool | Branches of attribute * ((value * tree) list)
*)

    (* Converts tree to a corresponding list of rules *)
    let tree_to_rules (t: tree) : rule list =
        let rec helper (t: tree) (r: rule) : rule list =
            match t with
            | Leaf(b) -> [{conditions = r.conditions; result = b}]
            | Branches(a, lst) -> List.concat (List.map ~f:(fun (v, subt) -> 
                helper subt {conditions = r.conditions @ [(a,v)]; 
                             result = false}) lst)
        in helper t {conditions = []; result = false}

    (* Sorts list of rules by accuracy *)
    let sort (rflst: (rule * float) list) : (rule * float) list =
        List.sort ~cmp:(fun (_, a) (_, b) -> a -. b) rflst
        
    (* Determines lower bound of 95% accuracy interval of rule. Assumes 
     * that some datapoints will be able to be predicted by rule. *)
    let get_accuracy (r: rule) (data: datapoint list) : float =
        let satisfied = List.fold_left ~init:data ~f:(fun a b -> List.filter 
                ~f:(fun x -> List.mem b x.attributes) a) r.conditions in
        let correct = List.filter ~f:(fun x -> x.classification = r.result) 
                satisfied in
        let n = Float.of_int (List.length satisfied) in
        let acc = (Float.of_int (List.length correct)) /. n in
        acc -. 1.96 *. sqrt (acc *. (1. -. acc) /. n)
        
    (* Prunes each rule, removing conditions one at a time to improve
     * estimated accuracy *)
    let prune (r: rule) (data: datapoint list) : rule * float = 
        let rec helper ((r, a): rule * float) (data: datapoint list) : 
          rule * float =
            let rlist = List.map ~f:(fun x -> {conditions = List.filter 
                    ~f:(fun y -> not (y = x)) r.conditions; result = r.result}) 
                    r.conditions in
            let sorted = sort (List.map ~f:(fun x -> 
                    (x, get_accuracy x data)) rlist) in
            match sorted with
            | [] -> (r, a)
            | hd::_ -> if snd hd < a then (r, a) else helper hd data
        in helper (r, get_accuracy r data) data
    
    (* Builds tree using ID3 algorithm then prunes the tree to avoid
     * overfitting the data and improve accuracy. *)
    let build (data: datapoint list) : rule list = 
        let rlst = tree_to_rules (T.build data) in
        let sorted = sort (List.map ~f:(fun x -> prune x data) rlst) in
        List.map ~f:fst sorted
    
    let predict (rlst: rule list) (attributes: attr_list) : outcome = 
        raise INCOMPLETE
    
    let print_tree (t: tree) : unit = ()
end

**************************************************************************)

module InfoGainID3Tree = ID3Tree(InfoGain)


