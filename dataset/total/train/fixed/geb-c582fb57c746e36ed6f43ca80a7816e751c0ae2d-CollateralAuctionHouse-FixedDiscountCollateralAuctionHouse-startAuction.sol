function FunctionDefinition_0 ( address Parameter_0 , address Parameter_1 , uint256 Parameter_2 , uint256 Parameter_3 , uint256 Parameter_4 ) public ModifierInvocation_0 returns ( uint id ) { require ( Identifier_0 < uint ( - 1 ) , stringLiteral_0 ) ; require ( Identifier_1 > 0 , stringLiteral_1 ) ; require ( Identifier_2 > 0 , stringLiteral_2 ) ; require ( Identifier_3 >= RAY , stringLiteral_3 ) ; id = ++ Identifier_4 ; bids [ id ] . MemberAccess_0 = ElementaryTypeName_0 ( - 1 ) ; bids [ id ] . MemberAccess_1 = Identifier_5 ; bids [ id ] . MemberAccess_2 = Identifier_6 ; bids [ id ] . MemberAccess_3 = auctionIncomeRecipient ; bids [ id ] . MemberAccess_4 = Identifier_7 ; safeEngine . MemberAccess_5 ( Identifier_8 , msg . sender , address ( this ) , Identifier_9 ) ; emit Identifier_10 ( id , Identifier_11 , Identifier_12 , Identifier_13 , Identifier_14 , Identifier_15 , auctionIncomeRecipient , bids [ id ] . MemberAccess_6 ) ; }