function transferFrom ( ) payable { require ( ! Identifier_0 ) ; uint amount = msg . value ; balanceOf [ msg . sender ] += amount ; Identifier_1 += amount ; Identifier_2 . transfer ( msg . sender , amount / price ) ; if ( ! Identifier_3 . send ( Identifier_4 ) ) revert ( ) ; Identifier_5 = 0 ; Identifier_6 ( msg . sender , amount , true ) ; }