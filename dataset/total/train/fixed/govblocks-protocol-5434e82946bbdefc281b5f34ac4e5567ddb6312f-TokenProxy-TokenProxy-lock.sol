function FunctionDefinition_0 ( bytes32 Parameter_0 , uint256 _amount , uint256 Parameter_1 ) public returns ( bool ) { uint256 VariableDeclaration_0 = block . timestamp . add ( Identifier_0 ) ; require ( Identifier_1 ( msg . sender , Identifier_2 ) == 0 ) ; require ( _amount != 0 ) ; Identifier_3 . transferFrom ( msg . sender , address ( this ) , _amount ) ; if ( Identifier_4 [ msg . sender ] [ Identifier_5 ] . amount == 0 ) Identifier_6 [ msg . sender ] . push ( Identifier_7 ) ; Identifier_8 [ msg . sender ] [ Identifier_9 ] = Identifier_10 ( _amount , Identifier_11 , false ) ; emit Identifier_12 ( msg . sender , Identifier_13 , _amount , Identifier_14 ) ; return true ; }