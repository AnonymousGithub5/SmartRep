function FunctionDefinition_0 ( address Parameter_0 , address Parameter_1 ) internal view returns ( uint256 Parameter_2 , uint256 Parameter_3 ) { if ( Identifier_0 != Identifier_1 ) { if ( Identifier_2 == Identifier_3 . MemberAccess_0 ) { if ( Identifier_4 == Identifier_5 ) { rate = Identifier_6 ; } else if ( Identifier_7 == Identifier_8 ) { rate = SafeMath . div ( 10 ** NumberLiteral_0 , Identifier_9 ) ; } else { ( bool VariableDeclaration_0 , bytes memory data ) = Identifier_10 . MemberAccess_1 ( abi . MemberAccess_2 ( stringLiteral_0 , Identifier_11 , Identifier_12 , 10 ** NumberLiteral_1 ) ) ; assembly { switch AssemblyExpression_0 case 0 { rate := 0 } default { rate := mload ( add ( AssemblyExpression_1 , DecimalNumber_0 ) ) } } } } else if ( Identifier_13 == Identifier_14 . MemberAccess_3 ) { return super . MemberAccess_4 ( Identifier_15 , Identifier_16 ) ; } else { rate = Identifier_17 [ Identifier_18 ] [ Identifier_19 ] ; } Identifier_20 = Identifier_21 ( Identifier_22 , Identifier_23 ) ; } else { rate = 10 ** 18 ; Identifier_24 = 10 ** 18 ; } }