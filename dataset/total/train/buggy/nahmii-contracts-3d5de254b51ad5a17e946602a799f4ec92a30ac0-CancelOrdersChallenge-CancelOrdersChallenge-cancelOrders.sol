function FunctionDefinition_0 ( UserDefinedTypeName_0 Parameter_0 ) public { require ( Identifier_0 != address ( 0 ) , stringLiteral_0 ) ; for ( uint256 i = 0 ; i < Identifier_1 . length ; i ++ ) { require ( msg . sender == Identifier_2 [ i ] . MemberAccess_0 ) ; require ( validator . MemberAccess_1 ( Identifier_3 [ i ] , owner ) ) ; } for ( uint256 VariableDeclaration_0 = 0 ; j < Identifier_4 . length ; j ++ ) { Identifier_5 [ msg . sender ] [ Identifier_6 [ j ] . MemberAccess_2 . MemberAccess_3 . MemberAccess_4 ] = true ; Identifier_7 [ msg . sender ] . push ( Identifier_8 [ j ] ) ; Identifier_9 [ msg . sender ] [ Identifier_10 [ j ] . MemberAccess_5 . MemberAccess_6 . MemberAccess_7 ] = Identifier_11 [ msg . sender ] . length - 1 ; } Identifier_12 [ msg . sender ] = block . timestamp . add ( Identifier_13 . MemberAccess_8 ( ) ) ; emit Identifier_14 ( Identifier_15 , msg . sender ) ; }