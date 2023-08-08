Terminals: `t`
Non-terminals: t

t?: one or more
t*: zero or more

AttributeDict := `{` (STRING `:` Parsable,) \* `}`

Module := `{`Function*`}` 
Function := `def` IDENT `(`Arg`:` Type (`,` Arg`:` Type)\*`)` `{` Stmt* `}`
Stmt := CoreStmt (`#` AttributeDict | $\epsilon$ )
CoreStmt := DeclareStmt 
		| BufferStoreStmt 
		| AssignStmt 
		| ReturnStmt 
		| LetStmt 
		| ForStmt 
		| ForMappingStmt
		| WhileStmt
		| BreakStmt
		| ContinueStmt
		| IfStmt
		| AssertStmt
		| BlackBoxStmt
		| SeqStmt

DeclareStmt := `decl` IDENT `:` Type ( `=` Expr | $\epsilon$ )`;`
BufferStoreStmt := IDENT `[`Expr`]` `=` Expr `;`
AssignStmt := IDENT `=` Expr `;`
