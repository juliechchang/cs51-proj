(** Dennis Lu, Julie Chang **)
(** CS51 Final Project **)
(** DTrees.ml contains modules for the ID3 decision tree algorithm 
    and rule post-pruning algorithm **)

open Core.Std
open Types
open Heuristics

module type DTREE =
sig
    val build : datapoint list -> (attribute * (value list)) list -> tree
    val predict : tree -> attr_list -> outcome
    val print_tree : tree -> out_channel -> unit
end

module ID3Tree (H: HEURISTIC) : DTREE =
struct
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
    let partition (data: datapoint list) (attr: attribute) (vals: value list) :
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
        in helper data (List.map ~f:(fun x -> ([], x)) vals)

    (* Prints tree in a human-readable form *)
    let print_tree (t: tree) (c: out_channel) : unit =
      (* Returns string representation of an attribute-value list *)
      let rec branches_to_string (lst: (value * tree) list) : string =
        match lst with
        | [] -> ""
        | [(v,t)] -> "(" ^ v ^ ", " ^ (tree_to_string t) ^ ")"
        | (v,t)::hd2::tl -> "(" ^ v ^ ", " ^ (tree_to_string t) ^ "), " ^ 
                            branches_to_string (hd2::tl)
      and tree_to_string (t: tree) : string =
        match t with
        | Leaf b -> "Leaf " ^ (string_of_bool b)
        | Branches (a,lst) -> "Branches (" ^ a ^ ", [" ^ 
            (branches_to_string lst) ^ "])"
      in fprintf c "%s\n" ((tree_to_string t) ^ "\n")
      
    (* Get list of all values corresponding to a particular attribute *)
    let get_values (poss: (attribute * (value list)) list) (a: attribute) :
      value list =
        match List.find ~f:(fun x -> fst x = a) poss with
        | None -> failwith "Attribute not found"
        | Some (_, vlst) -> vlst 
    
    (* Build decision tree given a dataset. First check if all outcomes
     * are the same: if they are, then we just use that outcome; if they
     * aren't, then we check if there are any attributes left. If there
     * aren't, then we use the most common outcome. Otherwise, we
     * compute the heuristic for every attribute, using the largest
     * heuristic to partition the dataset by the corresponding value.
     * We remove the attribute-value pair from each data point and 
     * recurisvely build the rest of the tree. *)
    let rec build (data: datapoint list) 
      (poss: (attribute * (value list)) list) : tree = 
        if same_outcomes data then
            match data with
            | [] -> raise NODATA
            | hd::_ -> Leaf(hd.classification)
        else if no_attributes data then Leaf(most_common data)
        else
            let heuristic_list = List.map ~f:(fun x -> 
                (x, H.compute data (partition data x (get_values poss x))))
                (get_attributes data) in
            let next_attribute = fst (List.fold_left ~init:("", -1.) 
                ~f:(fun a b -> if snd b >= snd a then b else a) 
                heuristic_list) in
            let values = get_values poss next_attribute in
            let parts = partition data next_attribute values in
            Branches(next_attribute, List.map 
                ~f:(fun (a, b) -> if a = [] then (b, Leaf(most_common data))
                                  else (b, build a poss))
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
end


module C45Tree (T: DTREE) = 
struct
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
        List.rev (List.sort ~cmp:(fun (_, a) (_, b) -> 
            if a -. b < 0. then (-1) else if a -. b = 0. then 0 else 1) 
            rflst)
    
    (* Determine whether datapoint satifies conditions in rule *)
    let satisfy (r: rule) (attrs: attr_list) : bool =
        List.fold_left ~init:(true) ~f:(fun a b -> a && List.mem attrs b)
            r.conditions
        
    (* Determines lower bound of 95% accuracy interval of rule. Assumes 
     * that some datapoints will be able to be predicted by rule. *)
    let get_accuracy (r: rule) (data: datapoint list) : float =
        let satisfied = List.filter ~f:(fun x -> satisfy r x.attributes) 
                data in
        let correct = List.filter ~f:(fun x -> x.classification = r.result) 
                satisfied in
        let n = Float.of_int (List.length satisfied) in
        if n = 0. then 0.
        else
            let acc = (Float.of_int (List.length correct)) /. n in
            max (acc -. 1.96 *. sqrt (acc *. (1. -. acc) /. n)) 0.
        
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
        
    (* Returns string representation of an attribute-value list *)
    let rec attrs_to_string (attrs: attr_list) : string =
        match attrs with
        | [] -> ""
        | [(a,v)] -> "(" ^ a ^ ", " ^ v ^ ")"
        | (a,v)::hd2::tl -> "(" ^ a ^ ", " ^ v ^ "), " ^ 
                            attrs_to_string (hd2::tl)
    
    (* Prints rules in human-readable form *)
    let print_rules (rlst: rule list) (c: out_channel): unit = 
        let rule_to_string (r: rule) : string = 
            let attrs_string = attrs_to_string r.conditions in
            "{conditions: [" ^ attrs_string ^ "]; result: " ^ 
                (string_of_bool r.result) ^ "}\n"
        in List.iter rlst ~f:(fun x -> fprintf c "%s" (rule_to_string x))
    
    (* Builds tree using ID3 algorithm then prunes the tree to avoid
     * overfitting the data and improve accuracy. *)
    let build (data: datapoint list) (poss: (attribute * (value list)) list) :
      rule list = 
        let rlst = tree_to_rules (T.build data poss) in
        let sorted = sort (List.map ~f:(fun x -> prune x data) rlst) in
        List.map ~f:fst sorted
    
    (* Predicts outcome of an attribute-value list using a list of rules *)
    let predict (rlst: rule list) (attributes: attr_list) : outcome = 
        match List.find rlst ~f:(fun x -> satisfy x attributes) with
        | None -> failwith "Rule list not comprehensive"
        | Some r -> r.result
end




