function FunctionDefinition_0 ( address Parameter_0 , address token , address Parameter_1 ) public returns ( uint Parameter_2 ) { uint VariableDeclaration_0 = Identifier_0 ( Identifier_1 ( token ) ) . balanceOf ( address ( this ) ) ; uint VariableDeclaration_1 = iUTILS ( Identifier_2 ( ) ) . MemberAccess_0 ( Identifier_3 , Identifier_4 ( Identifier_5 ( token ) ) . totalSupply ( ) , Identifier_6 [ token ] [ address ( this ) ] ) ; Identifier_7 ( Identifier_8 ( token ) ) . burn ( Identifier_9 ) ; Identifier_10 [ token ] [ address ( this ) ] -= Identifier_11 ; Identifier_12 [ token ] -= Identifier_13 ; Identifier_14 = iUTILS ( Identifier_15 ( ) ) . MemberAccess_1 ( Identifier_16 , Identifier_17 [ token ] , Identifier_18 [ token ] ) ; emit Identifier_19 ( member , base , Identifier_20 , token , 0 , Identifier_21 , Identifier_22 [ token ] ) ; Identifier_23 [ token ] -= Identifier_24 ; Identifier_25 ( base , Identifier_26 , member ) ; return Identifier_27 ; }