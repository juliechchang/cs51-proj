(** Dennis Lu, Julie Chang **)
(** CS51 Final Project **)
(** Types.ml defines types used in the decision tree learning algorithms **)

open Core.Std

exception INCOMPLETE
exception NODATA

type attribute = string
type value = string
type outcome = bool

type attr_list = (attribute * value) list
type datapoint = {attributes: attr_list; classification: outcome}
type rule = {conditions: attr_list; result: outcome}

(** Tree **)
type tree = Leaf of bool | Branches of attribute * ((value * tree) list)
