function FunctionDefinition_0 ( uint256 Parameter_0 , uint256 Parameter_1 , address to , uint256 Parameter_2 ) external returns ( uint256 memory Parameter_3 ) { Identifier_0 . safeTransferFrom ( Identifier_1 , msg . sender , address ( this ) , Identifier_2 ) ; address memory VariableDeclaration_0 = new address ( 2 ) ; path [ 0 ] = Identifier_3 ; path [ 1 ] = Identifier_4 ; Identifier_5 ( Identifier_6 ) . withdraw ( 0 ) ; uint256 memory VariableDeclaration_1 = Identifier_7 ( Identifier_8 ) . MemberAccess_0 ( Identifier_9 ( Identifier_10 ) , Identifier_11 , path , address ( this ) , Identifier_12 ) ; Identifier_13 ( Identifier_14 ) . withdraw ( Identifier_15 [ 1 ] ) ; amounts = new uint256 ( 2 ) ; amounts [ 0 ] = Identifier_16 [ 0 ] ; amounts [ 1 ] = address ( this ) . balance ; Identifier_17 . MemberAccess_1 ( to , address ( this ) . balance ) ; if ( Identifier_18 > amounts ) { Identifier_19 . safeTransfer ( Identifier_20 , msg . sender , Identifier_21 . sub ( amounts [ 0 ] ) ) ; } }