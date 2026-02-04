#include "Types.r"

/* Resource IDs */
#define kPrefDialogID 128
#define kAlertID      129

/* 
   DLOG Layout:
   Top:    50
   Left:   20
   Bottom: 450
   Right:  480
*/

resource 'DLOG' (kPrefDialogID, "RunSmol Config") {
    {50, 20, 450, 480},
    dBoxProc, visible, noGoAway, 0x0, kPrefDialogID, "Run-Smol Config", centerMainScreen
};

resource 'DITL' (kPrefDialogID, "RunSmol Items") {
    {
        /* 1: Start Button */
        {340, 380, 360, 450}, Button { enabled, "Start" },
        
        /* 2: Quit Button */
        {340, 300, 360, 370}, Button { enabled, "Quit" },
        
        /* 3: Select Model Button */
        {15, 20, 35, 140},    Button { enabled, "Select Model..." },
        
        /* 4: Model Name Text */
        {18, 150, 34, 450},   StaticText { enabled, "No Model Selected" },
        
        /* 5: Chat Mode Radio */
        {50, 20, 70, 110},    RadioButton { enabled, "Chat" },
        
        /* 6: Generate Mode Radio */
        {50, 120, 70, 240},   RadioButton { enabled, "Generate" },
        
        /* 7: Reset Defaults Button */
        {340, 20, 360, 130},  Button { enabled, "Reset Defaults" },

        /* 8: LARGE INPUT TEXT (Prompt) */
        {100, 20, 180, 450},  EditText { enabled, "" }, 
        
        /* 9: Disable Prompt Checkbox */
        {185, 20, 205, 220},  CheckBox { enabled, "Disable / Empty Prompt" },

        /* 10: Temp Edit */
        {230, 65, 246, 115},  EditText { enabled, "0.8" },
        
        /* 11: Top_P Edit */
        {230, 175, 246, 225}, EditText { enabled, "0.9" },
        
        /* 12: Top_K Edit */
        {230, 285, 246, 335}, EditText { enabled, "40" },

        /* 13: Seed Edit */
        {265, 120, 281, 230}, EditText { enabled, "" },
        
        /* 14: Random Check */
        {265, 20, 285, 110},  CheckBox { enabled, "Random Seed" },

        /* IMPORTANT: StaticText items must be 'enabled' to render correctly in this context */
        /* 15: Input Box Label */
        {80, 20, 96, 300},    StaticText { enabled, "System Prompt (Chat) / Initial Text (Gen):" },
        
        /* 16: Temp Label */
        {233, 20, 249, 60},   StaticText { enabled, "Temp:" },
        
        /* 17: P Label */
        {233, 130, 249, 170}, StaticText { enabled, "Top P:" },
        
        /* 18: K Label */
        {233, 240, 249, 280}, StaticText { enabled, "Top K:" },
        
        /* 19: Separator/Note */
        {295, 20, 311, 450},  StaticText { enabled, "Note: Requires G4 (AltiVec) and ~64MB RAM." }
    }
};

resource 'ALRT' (kAlertID, "Pick Model Alert") {
    {80, 60, 180, 360},
    kAlertID,
    { OK, visible, sound1, OK, visible, sound1, OK, visible, sound1, OK, visible, sound1 },
    alertPositionMainScreen
};

resource 'DITL' (kAlertID, "Pick Model Items") {
    {
        {65, 220, 85, 280}, Button { enabled, "OK" },
        {10, 60, 60, 290},  StaticText { enabled, "Please select a model.bin file." },
        {10, 10, 42, 42},   Icon { disabled, 2 } 
    }
};
