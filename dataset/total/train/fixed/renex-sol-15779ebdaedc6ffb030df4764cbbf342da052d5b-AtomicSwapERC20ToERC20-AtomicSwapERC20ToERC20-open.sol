function FunctionDefinition_0 ( bytes32 Parameter_0 , uint256 Parameter_1 , address Parameter_2 , uint256 Parameter_3 , address Parameter_4 , address Parameter_5 ) public ModifierInvocation_0 ( Identifier_16 ) { ERC20 VariableDeclaration_0 = ERC20 ( Identifier_0 ) ; require ( Identifier_1 <= Identifier_2 . MemberAccess_0 ( msg . sender , address ( this ) ) ) ; require ( Identifier_3 . transferFrom ( msg . sender , address ( this ) , Identifier_4 ) ) ; UserDefinedTypeName_0 memory VariableDeclaration_1 = Identifier_5 ( { timestamp : now , openValue : Identifier_6 , openTrader : msg . sender , openContractAddress : Identifier_7 , closeValue : Identifier_8 , closeTrader : Identifier_9 , closeContractAddress : Identifier_10 } ) ; Identifier_11 [ Identifier_12 ] = swap ; Identifier_13 [ Identifier_14 ] = Identifier_15 . MemberAccess_1 ; }