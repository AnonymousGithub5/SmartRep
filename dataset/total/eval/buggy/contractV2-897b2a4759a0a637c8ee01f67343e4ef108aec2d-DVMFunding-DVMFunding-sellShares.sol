function FunctionDefinition_0 ( address to , uint256 Parameter_0 , bytes calldata data ) external ModifierInvocation_0 returns ( uint256 ) { require ( Identifier_0 . balanceOf ( msg . sender ) >= Identifier_1 , stringLiteral_0 ) ; ( uint256 VariableDeclaration_0 , uint256 VariableDeclaration_1 ) = Identifier_2 . MemberAccess_0 ( ) ; uint256 VariableDeclaration_2 = Identifier_3 . totalSupply ( ) ; Identifier_4 . burn ( msg . sender , Identifier_5 ) ; uint256 VariableDeclaration_3 = Identifier_6 . mul ( Identifier_7 ) . div ( Identifier_8 ) ; uint256 VariableDeclaration_4 = Identifier_9 . mul ( Identifier_10 ) . div ( Identifier_11 ) ; Identifier_12 . MemberAccess_1 ( to , Identifier_13 ) ; Identifier_14 . MemberAccess_2 ( to , Identifier_15 ) ; Identifier_16 . MemberAccess_3 ( ) ; if ( data . length > 0 ) Identifier_17 ( to , Identifier_18 , Identifier_19 , Identifier_20 , data ) ; }