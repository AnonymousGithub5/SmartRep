function FunctionDefinition_0 ( ) public { Identifier_0 ( ) ; uint VariableDeclaration_0 = Identifier_1 ( ) ; Identifier_2 ( address ( Identifier_3 ) , 10 * Identifier_4 ) ; Identifier_5 . MemberAccess_0 ( address ( Identifier_6 ) , 5 * Identifier_7 ) ; ( uint buyPrice , ) = pool . MemberAccess_1 ( symbol ) ; uint VariableDeclaration_1 = NumberLiteral_0 * Identifier_8 / 10 ; ( bool success , ) = address ( Identifier_9 ) . call ( abi . encodePacked ( Identifier_10 . MemberAccess_2 . selector , abi . encode ( symbol , Identifier_11 - 1 , Identifier_12 ) ) ) ; Assert . MemberAccess_3 ( success , stringLiteral_0 ) ; }