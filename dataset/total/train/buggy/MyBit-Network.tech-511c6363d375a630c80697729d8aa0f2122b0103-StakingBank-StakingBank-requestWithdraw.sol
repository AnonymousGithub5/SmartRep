function FunctionDefinition_0 ( bytes32 Parameter_0 ) external nonReentrant ModifierInvocation_0 whenNotPaused ModifierInvocation_1 ( Identifier_4 , msg . sender ) ModifierInvocation_2 ( Identifier_5 ) returns ( bool ) { require ( database . MemberAccess_0 ( keccak256 ( stringLiteral_0 ) ) == false ) ; Identifier_0 ( msg . sender , Identifier_1 ) ; uint VariableDeclaration_0 = database . uintStorage ( keccak256 ( stringLiteral_1 ) ) ; database . MemberAccess_1 ( keccak256 ( stringLiteral_2 ) , Identifier_2 . sub ( Identifier_3 ) ) ; database . MemberAccess_2 ( keccak256 ( stringLiteral_3 ) , true ) ; return true ; }