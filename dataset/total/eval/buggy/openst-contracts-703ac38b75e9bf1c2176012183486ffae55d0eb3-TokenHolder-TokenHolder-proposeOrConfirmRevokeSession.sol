function FunctionDefinition_0 ( bytes32 Parameter_0 , bool Parameter_1 ) public ModifierInvocation_0 ModifierInvocation_1 ( Identifier_18 ) returns ( bytes32 Parameter_2 ) { require ( Identifier_0 [ Identifier_1 ] != uint256 ( 0 ) , stringLiteral_0 ) ; Identifier_2 = keccak256 ( abi . encodePacked ( Identifier_3 , this , stringLiteral_1 ) ) ; if ( Identifier_4 ) { require ( Identifier_5 ( Identifier_6 ) == false , stringLiteral_2 ) ; Identifier_7 ( Identifier_8 ) ; } else { Identifier_9 ( Identifier_10 ) ; if ( Identifier_11 ( Identifier_12 ) ) { delete Identifier_13 [ Identifier_14 ] ; emit Identifier_15 ( msg . sender , Identifier_16 ) ; } } return Identifier_17 ; }