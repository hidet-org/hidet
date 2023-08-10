### Notes
Terminals: `t`
Non-terminals: t

t?: one or more
t*: zero or more

ALL_CAPS: Predefined terminal
- STRING: A string enclosed by `""`
- IDENT: A valid identifier in C/Python
- INT/FLOAT/STRING: A valid literal in Python

$\epsilon$ : Null
( t | $\epsilon$ ) : Optional

#### Minutiae
- `@` to prevent ambiguity with Call
- Every statement has an optional `#` IDENT, which encodes an optional attribute dict, the same attribute dict is defined in the top level module space
- We encode things that are not part of the grammar as an attribute, for example, layouts. This keeps everything relatively extensible

## Grammar
Value := STRING | INT | FLOAT | Dict | List | Tuple
DictItem := STRING `:` Value
Tuple := `(` ( Value | $\epsilon$ ) (`,` Value)\* `)`
List := `[` ( Value | $\epsilon$ ) (`,` Value)\* `]`
Dict := `{` ( DictItem | $\epsilon$ ) (`,` DictItem)\* `}`

AttributeName := `#` IDENT
Attribute := AttributeName `=` Dict `;`

DataType := `i1` | `i8` | `i16` | `i32` | `i64` | `u8` | `u16` | `u32` | `u64` | `f16` | `f32` | `f64` | `void` | `str`
PtrType := `~`DataType
TensorType := DataType`<` NAT (`,` NAT)\* (`,` AttributeName | $\epsilon$ ) `>`
TensorPtrType := `~`TensorType
Type := DataType | PtrType | TensorType | TensorPtrType

Module := `{` ( Function | Attribute )\* `}` 
Function := `def` IDENT `(` ( Arg`:` Type | $\epsilon$ ) (`,` Arg`:` Type)\* `)` `->` Type `{` Stmt* `}`
Stmt := CoreStmt ( AttributeName `;` | $\epsilon$ )
CoreStmt := DeclareStmt 
		| BufferStoreStmt 
		| AssignStmt 
		| ReturnStmt 
		| LetStmt 
		| ForStmt 
		| WhileStmt
		| BreakStmt
		| ContinueStmt
		| IfStmt
		| AssertStmt
		| BlackBoxStmt
		| AsmStmt

DeclareStmt := `decl` IDENT `:` Type ( `=` Expr | $\epsilon$ ) `;`
BufferStoreStmt := IDENT `[`Expr`]` `=` Expr `;`
AssignStmt := IDENT `=` Expr `;`
ReturnStmt := `return` Expr `;`
LetStmt := `let` IDENT `:` Type `=` Expr  ( `in` Stmt | $\epsilon$ ) `;`
ForStmt := `for` ( IDENT | `(` IDENT (`,` IDENT)\* `)` ) `in` Mapping `{` Stmt* `}`
WhileStmt := `while` Expr `{` Stmt* `}`
BreakStmt := `break` `;`
ContinueStmt := `continue` `;`
IfStmt := `if` Expr `{` Stmt* `}` ( `else` `{` Stmt* `}` | $\epsilon$ )
AssertStmt := `assert` Expr `;`
BlackBoxStmt := `BlackBox` `{` STRING `}`
AsmStmt := ( `volatile` | $\epsilon$ ) `asm` `{` `{` STRING `}` `{` IDENT (`,` IDENT) `}` `{` IDENT (`,` IDENT) `}` `}`


BinaryOp := `+` | `*` | `-` | `/` | `%` | `**` | `^` | `<<` | `>>` | `||` | `&&` | `==` | `!=` | `<` | `>` | `<=` | `>=`
UnaryOp := `-` | `!` 
Expr := `(` Expr BinaryOp Expr `)`
		| `(` UnaryOp Expr `)`
		| GetItem
		| TensorSlice
		| Call
		| Let
		| Cast
		| Constant
		| IfThenElse
		| Dereference
		| Address
		| Reference
		| IDENT

GetItem := Expr `[` Expr `]`
Slice := ( Expr | $\epsilon$ ) `:` ( Expr | $\epsilon$ )
TensorSlice := Expr `[` Slice (`,` Slice)\* `]`
Call := IDENT `(` ( Expr | $\epsilon$ ) (`,` Expr)\* `)`
Let := `let` `(` IDENT `:` Type `=` Expr `)` `in` `(` Expr `)`
Cast := IDENT `as` Type
Constant := BOOl | INT | FLOAT
IfThenElse := `if` Expr `then` `{` Expr `}` `else` `{` Expr `}`
Dereference := `deref` Expr
Address := `addr` Expr
Reference := `ref` Expr

Mapping := SpatialTaskMapping | RepeatTaskMapping | ComposedTaskMapping
SpatialTaskMapping := `@spatial` `(` `shape` `=` `[` INT (`,` INT)\* `]` `,` `ranks` `=` `[` INT (`,` INT)\* `]` `)`
RepeatTaskMapping := `@repeat` `(` `shape` `=` `[` INT (`,` INT)\* `]` `,` `ranks` `=` `[` INT (`,` INT)\* `]` (`,` AttributeName | $\epsilon$ ) `)`
ComposedTaskMapping := `@compose` `(` Mapping `,` Mapping `)`
